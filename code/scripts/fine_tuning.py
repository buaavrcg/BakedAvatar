import numpy as np
import torch
import torch.nn as nn
import tqdm
import time
import os
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from collections import deque

from utils.training_util import InfiniteSampler, seed_everything, log_value_dict, add_dict_to
from utils.misc_util import ensure_dir, find_latest_model_path, construct_class_by_name, Logger
from scripts.visualization import visualize_outputs


def fine_tuning_loop(
    rundir,
    seed,
    use_cpu,
    mesh_data_path,
    # Dataset
    dataset_class,
    data_dir,
    train_subdirs,
    val_subdirs,
    test_subdirs,
    img_res,
    train_subsample,
    val_subsample,
    test_subsample,
    dataset_args,
    # Model
    model_class,
    model_args,
    finetune_model_class,
    finetune_model_args,
    # Dataloader
    batch_size,
    num_workers,
    no_shuffle,
    # Loss
    finetune_loss_class,
    finetune_loss_args,
    # Optimizer
    optim_class,
    optim_class_input,
    optim_args,
    optim_args_input,
    optimize_expression,
    optimize_pose,
    optimize_camera,
    # Scheduler
    scheduler_class,
    scheduler_class_input,
    scheduler_args,
    scheduler_args_input,
    # Training hyperparameters
    iterations,
    lr,
    lr_input,
    weight_decay,
    clip_grad_norm,
    # Logging
    log_it,
    show_it,
    save_it,
    eval_it,
    export_it,
    avg_loss_it,
    # Visualization
    vis_args,
    **kwargs,
):
    accel = Accelerator(
        cpu=use_cpu,
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=False)],
    )
    seed_everything(seed + accel.process_index)  # set seed
    if accel.is_local_main_process:
        output_dir = os.path.join(rundir, "finetune_output")
        original_checkpoints_dir = os.path.join(rundir, "checkpoints")
        checkpoints_dir = os.path.join(rundir, "finetune_checkpoints")
        export_dir = os.path.join(rundir, "finetune_mesh_data")
        tb_logger = SummaryWriter(os.path.join(rundir, "log"))
        Logger(os.path.join(rundir, "funetuning_log.txt"), "w+")
        ensure_dir(output_dir)
        ensure_dir(checkpoints_dir)
        ensure_dir(export_dir)
    accel.wait_for_everyone()

    # build train and validation dataset
    train_dataset = construct_class_by_name(class_name=dataset_class,
                                            data_dir=data_dir,
                                            sub_dirs=train_subdirs,
                                            img_res=img_res,
                                            num_rays=-1,
                                            subsample=train_subsample,
                                            use_semantics=finetune_loss_args.get(
                                                "use_semantic", False),
                                            no_gt=False,
                                            **dataset_args)
    accel.print(f"Loaded {len(train_dataset)} training frames from {data_dir}/{train_subdirs}.")
    val_dataset = construct_class_by_name(class_name=dataset_class,
                                          data_dir=data_dir,
                                          sub_dirs=val_subdirs,
                                          img_res=img_res,
                                          num_rays=-1,
                                          subsample=val_subsample,
                                          use_semantics=finetune_loss_args.get(
                                              "use_semantic", False),
                                          no_gt=False,
                                          **dataset_args)
    accel.print(f"Loaded {len(val_dataset)} validation frames from {data_dir}/{val_subdirs}.")
    train_loader = DataLoader(train_dataset,
                              batch_size,
                              sampler=InfiniteSampler(train_dataset, accel.local_process_index,
                                                      accel.num_processes, not no_shuffle, seed),
                              drop_last=False,
                              num_workers=num_workers,
                              persistent_workers=True,
                              pin_memory=True)
    assert len(val_dataset) % accel.num_processes == 0, \
        f"val dataset size not divisible by num processes"
    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            drop_last=True,
                            num_workers=num_workers)

    # build model and loss
    model = construct_class_by_name(class_name=finetune_model_class,
                                    shape_params=train_dataset.get_shape_params(),
                                    canonical_exp=train_dataset.get_mean_expression(),
                                    **model_args,
                                    **finetune_model_args)
    model.load_mesh_data(torch.load(mesh_data_path))
    loss = construct_class_by_name(class_name=finetune_loss_class, **finetune_loss_args)

    # build optimizer and scheduler
    optimizer = construct_class_by_name(model.parameters(),
                                        class_name=optim_class,
                                        lr=lr,
                                        weight_decay=weight_decay,
                                        **optim_args)
    scheduler = construct_class_by_name(optimizer, class_name=scheduler_class, **scheduler_args)
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    accel.print(f"Model parameters total: {total_num}, trainable: {trainable_num}")

    # build (optional) optimizable input parameters
    optimize_inputs = optimize_expression or optimize_pose or optimize_camera
    if optimize_inputs:
        input_params = nn.ModuleDict()
        num_training_frames = len(train_dataset)
        if optimize_expression:
            init_exp = train_dataset.get_expression_params()  # tracked expression [N, 50]
            init_exp = torch.cat(
                [init_exp,
                 torch.zeros(init_exp.shape[0], model_args["dim_expression"] - 50)], 1)
            expression = nn.Embedding(num_training_frames,
                                      model_args["dim_expression"],
                                      _weight=init_exp)
            input_params.add_module("expression", expression)
        if optimize_pose:
            init_pose = train_dataset.get_pose_params()  # tracked pose [N, 15]
            pose = nn.Embedding(num_training_frames, 15, _weight=init_pose)
            input_params.add_module("pose", pose)
        if optimize_camera:
            init_extrinsic = train_dataset.get_extrinsic_params()  # tracked extrinsic [N, 4, 4]
            cam_trans = nn.Embedding(num_training_frames, 3, _weight=init_extrinsic[:, :3, 3])
            input_params.add_module("cam_trans", cam_trans)

        optimizer_input = construct_class_by_name(input_params.parameters(),
                                                  class_name=optim_class_input,
                                                  lr=lr_input,
                                                  **optim_args_input)
        scheduler_input = construct_class_by_name(optimizer_input,
                                                  class_name=scheduler_class_input,
                                                  **scheduler_args_input)
        total_num = sum(p.numel() for p in input_params.parameters())
        trainable_num = sum(p.numel() for p in input_params.parameters() if p.requires_grad)
        accel.print(f"Input parameters total: {total_num}, trainable: {trainable_num}")

    # resume training or start from scratch
    last_ckpt_dir = find_latest_model_path(checkpoints_dir)
    if last_ckpt_dir:
        model_state = torch.load(os.path.join(last_ckpt_dir, "model.pth"), accel.device)
        model.load_state_dict(model_state['model'], strict=True)
        it, sample_count = model_state['it'], model_state['sample_count']

        optim_state = torch.load(os.path.join(last_ckpt_dir, "optimizer.pth"), accel.device)
        optimizer.load_state_dict(optim_state['optimizer'])
        scheduler.load_state_dict(optim_state['scheduler'])
        del model_state, optim_state

        if optimize_inputs:
            input_state, optim_input_state = None, None
            try:
                input_state = torch.load(os.path.join(last_ckpt_dir, "input.pth"), accel.device)
                input_params.load_state_dict(input_state)
                try:
                    optim_input_state = torch.load(
                        os.path.join(last_ckpt_dir, "optimizer_input.pth"), accel.device)
                    optimizer_input.load_state_dict(optim_input_state['optimizer'])
                    scheduler_input.load_state_dict(optim_input_state['scheduler'])
                except:
                    accel.print("Mismatched optimizer state for input parameters.")
            except:
                accel.print("Mismatched state for input parameters.")
            del input_state, optim_input_state

        accel.print(f'Loaded from: {last_ckpt_dir}')
    else:
        it, sample_count = 0, 0
        # Try load initial input params from original model checkpoint
        load_ckpt_dir = find_latest_model_path(original_checkpoints_dir)
        original_input_ckpt_path = os.path.join(load_ckpt_dir, "input.pth")
        if optimize_inputs and os.path.exists(original_input_ckpt_path):
            input_state = None
            try:
                input_state = torch.load(original_input_ckpt_path, accel.device)
                input_params.load_state_dict(input_state)
                accel.print(f"Loaded initial input params from {original_input_ckpt_path}")
            except:
                pass
            del input_state

    # accelerate model training
    model, loss, train_loader, val_loader, optimizer, scheduler = \
        accel.prepare(model, loss, train_loader, val_loader, optimizer, scheduler)
    if optimize_inputs:
        input_params, optimizer_input, scheduler_input = \
            accel.prepare(input_params, optimizer_input, scheduler_input)

    # start training
    accel.print(f'Start training from iteration {it}, ksample {sample_count / 1000: .3f}')
    last_it, last_time = it, time.time()
    avg_loss_dict = {}
    model.train()
    if optimize_inputs:
        input_params.train()

    def replace_optimizable_inputs(inputs, targets, fixed_id=None):
        """replace model inputs from optimizable parameters if needed"""
        ids = torch.full_like(inputs['id'], fixed_id) if fixed_id else inputs['id']
        if optimize_expression:
            targets['expression'] = inputs['expression']
            inputs['expression'] = input_params['expression'](ids)
        if optimize_pose:
            targets['pose'] = inputs['pose']
            inputs['pose'] = input_params['pose'](ids)
        if optimize_camera:
            targets['extrinsic'] = inputs['extrinsic']
            inputs['extrinsic'][:, :3, 3] = input_params['cam_trans'](ids)

    # infinite training loop
    for inputs, targets in train_loader:
        epoch = sample_count / len(train_dataset)

        # model forward
        optimizer.zero_grad()
        if optimize_inputs:
            optimizer_input.zero_grad()
            replace_optimizable_inputs(inputs, targets)
        outputs = model(inputs)
        loss_total, loss_dict = loss(outputs, targets)

        # update model parameters
        accel.backward(loss_total)
        if clip_grad_norm is not None:
            accel.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        optimizer.step()
        scheduler.step()
        if optimize_inputs:
            optimizer_input.step()
            scheduler_input.step()

        # update running average loss
        loss_dict = accel.gather(loss_dict)
        for key, value in loss_dict.items():
            if not key in avg_loss_dict:
                avg_loss_dict[key] = deque(maxlen=avg_loss_it)
            avg_loss_dict[key].append(value.item())
        for key, value in avg_loss_dict.items():
            loss_dict[key] = np.mean(avg_loss_dict[key])

        # logging
        if it % log_it == 0 and accel.is_local_main_process:
            log_value_dict(tb_logger, 'finetune', loss_dict, it)
        if it % show_it == 0 and accel.is_local_main_process:
            elasped = time.time() - last_time
            num_it = it - last_it
            speed = num_it / elasped
            log_value_dict(
                tb_logger, 'running_stat_finetune', {
                    'sample_count': sample_count,
                    'epoch': epoch,
                    'elasped_seconds': elasped,
                    'it/s': speed,
                    'sample/s': speed * batch_size * accel.num_processes,
                }, it)
            log_value_dict(tb_logger, 'learning_rate',
                           {'model_finetune': scheduler.get_last_lr()[0]} |
                           ({
                               'input_params_finetune': scheduler_input.get_last_lr()[0]
                           } if optimize_inputs else {}), it)
            loss_total = loss_dict.pop('loss')
            accel.print("".join([
                f"[{it:07d}][{epoch:.2f}][{elasped:.2f}s][{speed:.2f}it/s]",
                f" total: {loss_total:.4f}",
                *list(f", {n[5:] if n[:4] == 'loss' else n}: {v:.4f}"
                      for n, v in sorted(loss_dict.items())),
            ]))
            last_it, last_time = it, time.time()

        # checkpoint saving
        if it % save_it == 0 and accel.is_local_main_process:
            ckpt_dir = os.path.join(checkpoints_dir, f'iter_{it}')
            ensure_dir(ckpt_dir)
            torch.save(
                {
                    'it': it,
                    'sample_count': sample_count,
                    'epoch': epoch,
                    'model': accel.get_state_dict(model),
                }, os.path.join(ckpt_dir, "model.pth"))
            torch.save({
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
            }, os.path.join(ckpt_dir, "optimizer.pth"))
            if optimize_inputs:
                torch.save(input_params.state_dict(), os.path.join(ckpt_dir, "input.pth"))
                torch.save(
                    {
                        'optimizer': optimizer_input.state_dict(),
                        'scheduler': scheduler_input.state_dict(),
                    }, os.path.join(ckpt_dir, "optimizer_input.pth"))
            accel.print(f'Saved checkpoint at {it} iter to: {ckpt_dir}')

        # evaluate model with validation dataset and save results
        if it % eval_it == 0 and len(val_dataset) > 0:
            eval_start_time = time.time()
            model.eval()
            if optimize_inputs:
                input_params.eval()

            val_loss_dict = {}
            val_num_batches = torch.tensor([0], dtype=torch.long)
            for val_inputs, val_targets in tqdm.tqdm(val_loader,
                                                     desc='eval progress',
                                                     disable=not accel.is_local_main_process):
                with torch.no_grad():
                    val_outputs = model(val_inputs)
                    _, loss_dict = loss(val_outputs, val_targets)
                add_dict_to(val_loss_dict, loss_dict)
                val_num_batches.add_(1)

                # save visualization result
                visualize_outputs(output_dir, val_inputs, val_outputs, val_targets, it, **vis_args)

            # gather loss dicts from all processes
            val_loss_dict['num_batches'] = val_num_batches.to(accel.device)
            val_loss_dict = accel.gather(val_loss_dict)

            if accel.is_local_main_process:
                # average validation losses
                val_num_batches = torch.sum(val_loss_dict.pop('num_batches')).item()
                for k in val_loss_dict:
                    val_loss_dict[k] = torch.sum(val_loss_dict[k]).item() / val_num_batches

                # log validation results
                eval_elapsed = time.time() - eval_start_time
                num_eval_samples = val_num_batches * 1  # validation batch size = 1
                log_value_dict(tb_logger, 'finetune_validation', val_loss_dict, it)
                loss_total = val_loss_dict.pop('loss')
                accel.print("".join([
                    f"[validation {num_eval_samples} samples][{epoch:.2f}][{eval_elapsed:.2f}s]",
                    f" total: {loss_total:.4f}",
                    *list(f", {n[5:] if n[:4] == 'loss' else n}: {v:.4f}"
                          for n, v in sorted(val_loss_dict.items())),
                ]))
                last_time += eval_elapsed

            model.train()
            if optimize_inputs:
                input_params.train()

        # export mesh data to export dir
        if it % export_it == 0 and accel.is_local_main_process:
            export_path = os.path.join(export_dir, f'iter_{it}', 'mesh_data.pkl')
            ensure_dir(os.path.dirname(export_path), False)
            torch.save(model.export_mesh_data(), export_path)
            accel.print(f'Exported mesh data to {export_path}.')

        it += 1
        sample_count += batch_size * accel.num_processes
        if it > iterations:
            break
