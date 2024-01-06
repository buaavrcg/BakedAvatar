import numpy as np
import torch
import torch.nn as nn
import tqdm
import time
import json
import os
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils.training_util import seed_everything
from utils.misc_util import ensure_dir, find_latest_model_path, construct_class_by_name, Logger
from utils.image_util import load_rgb_image
from scripts.visualization import visualize_outputs


def fitting_inputs(
    accel,
    model,
    dataset,
    checkpoints_dir,
    fit_epochs,
    lr_input,
    optim_class_input,
    optim_args_input,
    scheduler_class_input,
    scheduler_args_input,
    optimize_latent,
    optimize_expression,
    optimize_pose,
    optimize_camera,
):
    input_params = nn.ModuleDict()
    num_training_frames = len(dataset)
    if optimize_latent:
        frame_latent = nn.Embedding(num_training_frames, model.dim_frame_latent)
        nn.init.uniform_(frame_latent.weight, 0, 1)
        input_params.add_module("frame_latent", frame_latent)
    if optimize_expression:
        init_exp = dataset.get_expression_params()  # tracked expression [N, 50]
        init_exp = torch.cat(
            [init_exp, torch.zeros(init_exp.shape[0], model.dim_expression - 50)], 1)
        expression = nn.Embedding(num_training_frames, model.dim_expression, _weight=init_exp)
        input_params.add_module("expression", expression)
    if optimize_pose:
        init_pose = dataset.get_pose_params()  # tracked pose [N, 15]
        pose = nn.Embedding(num_training_frames, 15, _weight=init_pose)
        input_params.add_module("pose", pose)
    if optimize_camera:
        init_extrinsic = dataset.get_extrinsic_params()  # tracked extrinsic [N, 4, 4]
        cam_trans = nn.Embedding(num_training_frames, 3, _weight=init_extrinsic[:, :3, 3])
        input_params.add_module("cam_trans", cam_trans)
    input_params = accel.prepare(input_params)

    def replace_optimizable_inputs(inputs, targets):
        """replace model inputs from optimizable parameters if needed"""
        ids = inputs['id']
        if optimize_latent:
            inputs['frame_latent'] = input_params['frame_latent'](ids)
        if optimize_expression:
            targets['expression'] = inputs['expression']
            inputs['expression'] = input_params['expression'](ids)
        if optimize_pose:
            targets['pose'] = inputs['pose']
            inputs['pose'] = input_params['pose'](ids)
        if optimize_camera:
            targets['extrinsic'] = inputs['extrinsic']
            inputs['extrinsic'][:, :3, 3] = input_params['cam_trans'](ids)

    input_ckpt_path = os.path.join(checkpoints_dir, "test_input.pth")
    try:
        input_state = torch.load(input_ckpt_path, accel.device)
        input_params.load_state_dict(input_state, strict=True)
        accel.print(f"Loaded input checkpoint from {input_ckpt_path}")
        return replace_optimizable_inputs
    except:
        accel.print(f"Input checkpoint not found at {input_ckpt_path}, "
                    f"initializing from scratch.")

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    optimizer = construct_class_by_name(input_params.parameters(),
                                        class_name=optim_class_input,
                                        lr=lr_input,
                                        **optim_args_input)
    scheduler = construct_class_by_name(optimizer,
                                        class_name=scheduler_class_input,
                                        **scheduler_args_input)
    total_num = sum(p.numel() for p in input_params.parameters())
    trainable_num = sum(p.numel() for p in input_params.parameters() if p.requires_grad)
    accel.print(f"Input parameters total: {total_num}, trainable: {trainable_num}")
    dataloader, optimizer, scheduler = accel.prepare(dataloader, optimizer, scheduler)

    # Since we may optimize the camera pose, input expression and pose during training, 
    # which can cause mismatched camera translation between training and test sets. 
    # For a better comparison of metrics, here we find the input parameters for each test
    # set sample to find the best matched parameters.
    accel.print(f"Start fitting input parameters with {len(dataset)} samples...")
    last_time = time.time()
    for epoch in range(fit_epochs):
        loss_total = 0.0

        optimizer.zero_grad()
        for inputs, targets in tqdm.tqdm(dataloader,
                                         desc=f"Fitting ({epoch+1}/{fit_epochs})",
                                         disable=not accel.is_local_main_process):
            replace_optimizable_inputs(inputs, targets)
            outputs = model(inputs)
            mask_gt = targets['mask']
            loss = nn.functional.l1_loss(outputs['rgb'][mask_gt], targets['rgb'][mask_gt])
            loss_total += loss.detach().cpu().item()
            accel.backward(loss)
        optimizer.step()
        scheduler.step()

        elasped = time.time() - last_time
        loss_total /= len(dataset)
        accel.print(f"[{epoch:02d}][{elasped:.2f}s] loss: {loss_total:.7f}")
        last_time = time.time()

    torch.save(input_params.state_dict(), input_ckpt_path)
    accel.print(f'Saved input params to: {input_ckpt_path}')

    return replace_optimizable_inputs


def testing_loop(
    rundir,
    seed,
    use_cpu,
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
    use_finetune_model,
    finetune_model_class,
    finetune_model_args,
    mesh_data_path,
    # Dataloader
    test_batch_size,
    num_workers,
    # Testing
    test_iteration,
    eval_only,
    metric_only,
    use_cache,
    image_dir,
    image_offset,
    use_train_mode,
    save_mesh,
    save_rgb_with_alpha,
    save_rgb_only,
    output_dir_name,
    # Params fitting
    fit_epochs,
    fit_img_res,
    fit_lr,
    fit_num_rays,
    optim_class_input,
    optim_args_input,
    scheduler_class_input,
    scheduler_args_input,
    optimize_latent,
    optimize_expression,
    optimize_pose,
    optimize_camera,
    # Metric
    metric_class,
    metric_args,
    # Visualization
    vis_args,
    # Facial Reenactment
    reenact_data_dir,
    reenact_subdirs,
    reenact_subsample,
    reenact_delta_transfer,
    # Expression editing
    expression_offset,
    pose_offset,
):
    accel = Accelerator(cpu=use_cpu)
    seed_everything(seed + accel.process_index)  # set seed
    if accel.is_local_main_process:
        Logger(os.path.join(rundir, "testing_log.txt"), "w+")

    # Load dataset
    train_dataset = construct_class_by_name(class_name=dataset_class,
                                            data_dir=data_dir,
                                            sub_dirs=train_subdirs,
                                            img_res=img_res,
                                            num_rays=-1,
                                            subsample=train_subsample,
                                            use_semantics=False,
                                            no_gt=True,
                                            **dataset_args)
    if reenact_data_dir is not None and reenact_subdirs is not None:
        data_dir = reenact_data_dir
        test_subdirs = reenact_subdirs
        test_subsample = reenact_subsample
        eval_only = True
        train_loader = DataLoader(train_dataset,
                                  batch_size=test_batch_size,
                                  shuffle=False,
                                  drop_last=True)
        train_loader = accel.prepare(train_loader)
        train_inputs = next(iter(train_loader))[0]
    if expression_offset is not None or pose_offset is not None:
        eval_only = True
    test_dataset = construct_class_by_name(class_name=dataset_class,
                                           data_dir=data_dir,
                                           sub_dirs=test_subdirs,
                                           img_res=img_res,
                                           num_rays=-1,
                                           subsample=test_subsample,
                                           use_semantics=True,
                                           no_gt=False,
                                           **dataset_args)
    accel.print(f"Loaded {len(test_dataset)} testing frames from {data_dir}/{test_subdirs}.")
    assert len(test_dataset) % accel.num_processes == 0, \
        f"test dataset size not divisible by num processes"
    test_loader = DataLoader(test_dataset,
                             batch_size=test_batch_size,
                             shuffle=False,
                             drop_last=True,
                             num_workers=num_workers)
    test_loader = accel.prepare(test_loader)

    # Build model and metrics
    if image_dir is not None:
        model = None
    elif use_finetune_model:
        model = construct_class_by_name(class_name=finetune_model_class,
                                        shape_params=train_dataset.get_shape_params(),
                                        canonical_exp=train_dataset.get_mean_expression(),
                                        **model_args,
                                        **finetune_model_args)
        if mesh_data_path is None and test_iteration is not None:
            mesh_data_path = os.path.join(rundir, "finetune_mesh_data", f"iter_{test_iteration}",
                                          "mesh_data.pkl")
        assert os.path.exists(mesh_data_path), f"Invalid mesh data path {mesh_data_path}"
        model.load_mesh_data(torch.load(mesh_data_path))
        # Load model parameters
        if test_iteration is not None:
            checkpoints_dir = os.path.join(rundir, "finetune_checkpoints", f"iter_{test_iteration}")
            assert os.path.exists(checkpoints_dir), "No checkpoints found!"
            model_state = torch.load(os.path.join(checkpoints_dir, "model.pth"), accel.device)
            model.load_state_dict(model_state['model'], strict=True)
            it, sample_count = model_state['it'], model_state['sample_count']
            accel.print(
                f'Loaded checkpoint (iter {it}, samples {sample_count}) from: {checkpoints_dir}')
        else:
            checkpoints_dir = os.path.dirname(mesh_data_path)
            it = None
            accel.print(f'Loaded mesh data from: {mesh_data_path}')
        model = accel.prepare(model)
    else:
        model = construct_class_by_name(class_name=model_class,
                                        shape_params=train_dataset.get_shape_params(),
                                        canonical_exp=train_dataset.get_mean_expression(),
                                        **model_args)
        checkpoints_dir = os.path.join(rundir, "checkpoints")
        # Load model parameters
        if test_iteration is not None:
            checkpoints_dir = os.path.join(checkpoints_dir, f"iter_{test_iteration}")
        else:
            checkpoints_dir = find_latest_model_path(checkpoints_dir)
        assert os.path.exists(checkpoints_dir), "No checkpoints found!"
        model_state = torch.load(os.path.join(checkpoints_dir, "model.pth"), accel.device)
        model.load_state_dict(model_state['model'], strict=True)
        it, sample_count = model_state['it'], model_state['sample_count']
        accel.print(
            f'Loaded checkpoint (iter {it}, samples {sample_count}) from: {checkpoints_dir}')
        model = accel.prepare(model)

    metrics = None
    if not eval_only:
        metrics = construct_class_by_name(class_name=metric_class, **metric_args)
        metrics = accel.prepare(metrics)

    # fit input parameters
    optimize_inputs = (optimize_latent or optimize_expression or optimize_pose or
                       optimize_camera) and image_dir is None and reenact_data_dir is None and \
                       expression_offset is None and pose_offset is None and fit_epochs > 0
    if optimize_inputs:
        fit_img_res = fit_img_res if fit_img_res is not None else img_res
        fitting_dataset = construct_class_by_name(class_name=dataset_class,
                                                  data_dir=data_dir,
                                                  sub_dirs=test_subdirs,
                                                  img_res=fit_img_res,
                                                  num_rays=fit_num_rays,
                                                  subsample=test_subsample,
                                                  use_semantics=False,
                                                  no_gt=False,
                                                  **dataset_args)
        replace_optimizable_inputs = fitting_inputs(
            accel,
            model,
            fitting_dataset,
            checkpoints_dir,
            fit_epochs,
            fit_lr,
            optim_class_input,
            optim_args_input,
            scheduler_class_input,
            scheduler_args_input,
            optimize_latent,
            optimize_expression,
            optimize_pose,
            optimize_camera,
        )

    # start testing
    accel.print(f"Start testing with {len(test_dataset)} samples...")
    if image_dir is not None:
        use_cache = True
        accel.print(f"Using images from {image_dir}...")
        output_dir = os.path.dirname(image_dir)
    else:
        if use_train_mode:
            model.train()
        else:
            model.eval()

        if output_dir_name is not None:
            output_dir = os.path.join(rundir, output_dir_name)
        elif reenact_data_dir is not None:
            output_dir = os.path.join(rundir, "reenact_output", os.path.basename(reenact_data_dir))
        elif expression_offset is not None or pose_offset is not None:
            output_dir = os.path.join(rundir, "edit_output")
        else:
            output_dir = os.path.join(rundir, "test_output")
        output_dir = os.path.join(output_dir, "mesh_data" if it is None else f"iter_{it}")

        if reenact_delta_transfer:
            source_expression_mean = train_dataset.get_mean_expression().to(accel.device)
            target_expression_mean = test_dataset.get_mean_expression().to(accel.device)
        if expression_offset is not None:
            expression_offset = torch.tensor(expression_offset, device=accel.device)
            assert expression_offset.ndim == 1 and expression_offset.shape[0] == model.dim_expression
        if pose_offset is not None:
            pose_offset = torch.tensor(pose_offset, device=accel.device)
    ensure_dir(output_dir, False)
    all_metrics = {}

    total_inference_time = 0
    total_inference_batch = 0
    for inputs, targets in tqdm.tqdm(test_loader,
                                     desc="Testing",
                                     disable=not accel.is_local_main_process):
        # use camera extrinsic and intrinsic from trainset
        if reenact_data_dir is not None and reenact_subdirs is not None:
            inputs["intrinsic"] = train_inputs["intrinsic"]
            inputs["extrinsic"] = train_inputs["extrinsic"]
        if reenact_delta_transfer:
            delta_exp = inputs["expression"] - target_expression_mean
            inputs["expression"] = source_expression_mean + delta_exp
        if expression_offset is not None:
            inputs['expression'] += expression_offset
        if pose_offset is not None:
            inputs['pose'] += pose_offset

        sub_dirs = inputs["sub_dir"]
        sub_ids = inputs["sub_id"].cpu().numpy()
        if save_mesh:
            mesh_save_path = []
            for i in range(test_batch_size):
                mesh_dir = os.path.join(output_dir, sub_dirs[i], "mesh")
                ensure_dir(mesh_dir, False)
                mesh_save_path.append(os.path.join(mesh_dir, f"{sub_ids[i] + 1}.obj"))
            inputs['mesh_save_path'] = mesh_save_path
        cache_available = use_cache
        if use_cache:
            rgb_images = []
            for i in range(test_batch_size):
                cached_rgb_path = os.path.join(image_dir if image_dir is not None else output_dir,
                                               sub_dirs[i], "rgb",
                                               f"{sub_ids[i] + image_offset}.png")
                if os.path.exists(cached_rgb_path):
                    rgb_image = load_rgb_image(cached_rgb_path, inputs['img_res'][i].cpu().numpy())
                    rgb_images.append(rgb_image.reshape(-1, 3))
                elif image_dir is None:
                    cache_available = False
                    del rgb_images
                    break
                else:
                    assert 0, f"rgb image not found at {cached_rgb_path}"
            if cache_available:
                rgb_images = np.stack(rgb_images, axis=0)  # (B, H*W, C)

        if cache_available:
            outputs = {
                'rgb': torch.from_numpy(rgb_images).to(accel.device),
                'img_res': inputs['img_res'],
            }
        else:
            if optimize_inputs:
                replace_optimizable_inputs(inputs, targets)

            with accel.no_sync(model):
                tic = time.time_ns()
                with torch.no_grad():
                    outputs = model(inputs)
                toc = time.time_ns()
                total_inference_time += (toc - tic) / 1e9
                total_inference_batch += 1

        if not metric_only and not cache_available:
            img_res = inputs['img_res'].cpu().numpy()
            for i in range(test_batch_size):
                rgb_dir = os.path.join(output_dir, sub_dirs[i], "rgb")
                ensure_dir(rgb_dir, False)
                rgb_tensor = outputs['rgb'][i].reshape(*img_res[i], 3).permute(2, 0, 1)
                if save_rgb_with_alpha:
                    alpha_tensor = outputs['alpha'][i].reshape(*img_res[i], 1).permute(2, 0, 1)
                    rgb_tensor = torch.cat([rgb_tensor, alpha_tensor])
                save_image(rgb_tensor, os.path.join(rgb_dir, f"{sub_ids[i] + 1}.png"))
                if not save_rgb_only and expression_offset is None and pose_offset is None:
                    normal_dir = os.path.join(output_dir, sub_dirs[i], "normal")
                    ensure_dir(normal_dir, False)
                    normal_tensor = outputs['normal'][i, :, 0].reshape(*img_res[i],
                                                                       3).permute(2, 0, 1)
                    normal_tensor = (normal_tensor + 1) / 2
                    save_image(normal_tensor, os.path.join(normal_dir, f"{sub_ids[i] + 1}.png"))
            if not save_rgb_only and not use_cache and reenact_data_dir is None and \
                expression_offset is None and pose_offset is None:
                visualize_outputs(output_dir, inputs, outputs, targets, **vis_args)

        if not eval_only:
            with torch.no_grad():
                metric_dict = metrics(outputs, targets)
            metric_dict["id"] = inputs["id"]

            metric_dict = accel.gather(metric_dict)
            if accel.is_local_main_process:
                for k, v in metric_dict.items():
                    v = v.cpu()
                    if k in all_metrics:
                        all_metrics[k].append(v)
                    else:
                        all_metrics[k] = [v]

    if total_inference_batch > 0:
        avg_inference_time = total_inference_time / total_inference_batch / test_batch_size
        accel.print(f"Total inference time: {total_inference_time:.4f} s")
        accel.print(f"Average inference time per image: {avg_inference_time:.4f} s, "
                    f"fps: {1 / avg_inference_time:.6f}")
    else:
        accel.print("Evaluated metrics from cache.")

    if accel.is_local_main_process and not eval_only:
        for k, v in all_metrics.items():
            all_metrics[k] = torch.cat(v, dim=0)

        sorted_idx = torch.argsort(all_metrics.pop("id"))
        for k, v in all_metrics.items():
            all_metrics[k] = v[sorted_idx]

        with open(os.path.join(output_dir, "metrics.csv"), "w") as f:
            keys = list(all_metrics.keys())
            f.write(",".join(["sub_dir", "sub_id"] + keys) + "\n")
            for i in range(len(sorted_idx)):
                data, _ = test_dataset[i]
                f.write(",".join([data["sub_dir"], str(data["sub_id"].item())] +
                                 [f"{all_metrics[k][i].item():.9f}" for k in keys]) + "\n")

        avg_metrics = {}
        accel.print("Averate metrics:")
        for k, v in all_metrics.items():
            avg_metrics[k] = v.mean().item()
            accel.print(f"{k}: {avg_metrics[k]}")
        with open(os.path.join(output_dir, "avg_metrics.json"), "w") as f:
            json.dump(avg_metrics, f, indent=4)
