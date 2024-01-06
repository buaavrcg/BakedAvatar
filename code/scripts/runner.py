import configargparse
import sys
import os
import yaml

root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from utils.misc_util import ensure_dir, EasyDict


def build_base_parser() -> configargparse.ArgumentParser:
    p = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('-t',
          '--task',
          required=True,
          choices=['train', 'test', 'mesh_export', 'texture_export', 'fine_tuning'],
          help='Task to run')
    p.add('-c', '--config', is_config_file=True, help='Config file path')
    p.add('-r', '--rundir', required=True, help="Experiments run directory")
    p.add('--run_name', default='subject1', help="Name of this run")
    p.add('--seed', type=int, default=42, help="Random seed")
    p.add('--use_cpu', action='store_true', help="Use cpu only")

    # Dataset
    p.add('--dataset_class', default='dataset.real.RealVideoDataset')
    p.add('--data_dir', required=True, help="Dataset directory path")
    p.add('--train_subdirs', nargs='+', required=True, help="Train subdirectories")
    p.add('--val_subdirs', nargs='+', required=True, help="Validation subdirectories")
    p.add('--test_subdirs', nargs='+', required=True, help="Test subdirectories")
    p.add('--img_res', type=int, nargs=2, default=(256, 256), metavar=('width', 'height'))
    p.add('--train_subsample', type=yaml.safe_load, default=1, help="Subsample frames in trainset")
    p.add('--val_subsample', type=yaml.safe_load, default=1, help="Subsample frames in valset")
    p.add('--test_subsample', type=yaml.safe_load, default=1, help="Subsample frames in testset")
    p.add('--dataset_args', type=yaml.safe_load, default={}, help="Extra dataset arguments")

    # Model
    p.add('--model_class', default='model.dmmavatar.DMMavatar')
    p.add('--model_args', type=yaml.safe_load, default={}, help="Model arguments")

    # Dataloader
    p.add('--num_workers', type=int, default=4, help="Number of dataloader workers")

    # Visualization
    p.add('--vis_args', type=yaml.safe_load, default={}, help="Visualization arguments")

    return p


def add_train_arguments(p: configargparse.ArgParser) -> None:
    # Resume training
    p.add('--load_from', help="Load pretrained weights from path")
    p.add('--load_iteration', type=int, help="Iteration to load weights from (default as latest)")

    # Dataloader
    p.add('--batch_size', type=int, default=8, help="Number of samples per batch")
    p.add('--num_rays', type=int, default=2048, help="Number of rays per sample")
    p.add('--no_shuffle', action='store_true', help="Do not shuffle train dataset")

    # Loss
    p.add('--loss_class', default='model.loss.Loss')
    p.add('--loss_args', type=yaml.safe_load, default={}, help="Loss arguments")

    # Optimizer
    p.add('--optim_class', default='torch.optim.AdamW')
    p.add('--optim_class_input', default='torch.optim.Adam')
    p.add('--optim_args', type=yaml.safe_load, default={})
    p.add('--optim_args_input', type=yaml.safe_load, default={})
    p.add('--optimize_latent', action='store_true', help="Optimize per-frame latent codes")
    p.add('--optimize_expression', action='store_true', help="Optimize expression inputs")
    p.add('--optimize_pose', action='store_true', help="Optimize pose inputs")
    p.add('--optimize_camera', action='store_true', help="Optimize camera inputs")

    # Scheduler
    p.add('--scheduler_class', default='torch.optim.lr_scheduler.ConstantLR')
    p.add('--scheduler_class_input', default='torch.optim.lr_scheduler.ConstantLR')
    p.add('--scheduler_args', type=yaml.safe_load, default={})
    p.add('--scheduler_args_input', type=yaml.safe_load, default={})

    # Training hyperparameters
    p.add('--iterations', type=int, default=1000000, help="Total number of iterations")
    p.add('--lr', type=float, default=1e-3, help="Learning rate for model params")
    p.add('--lr_input', type=float, default=1e-3, help="Learning rate for inputs")
    p.add('--weight_decay', type=float, default=0, help="Weight decay for model params")
    p.add('--clip_grad_norm', type=float, help="Gradient clipping max norm")

    # Logging
    p.add('--log_it', type=int, default=20, help="Num iterations to log")
    p.add('--show_it', type=int, default=100, help="Num iterations to display")
    p.add('--save_it', type=int, default=5000, help="Num iterations to save snapshot")
    p.add('--eval_it', type=int, default=5000, help="Num iterations to evaluate result")
    p.add('--avg_loss_it', type=int, default=100, help="Num iterations for average loss")


def add_test_arguments(p: configargparse.ArgParser) -> None:
    # Model
    p.add('--use_finetune_model', action='store_true', help="Use fine tuning model for testing")
    p.add('--finetune_model_class', default='model.rasterization.RasterizationModel')
    p.add('--finetune_model_args', type=yaml.safe_load, default={}, help="Model arguments")
    p.add('--mesh_data_path', help="The path of mesh data json file")

    # Dataloader
    p.add('--test_batch_size', type=int, default=1, help="Number of images per batch")

    # Testing
    p.add('--test_iteration', type=int, help="Iteration to load weights from (default as latest)")
    p.add('--eval_only', action='store_true', help="Evaluate result without calculating metrics")
    p.add('--metric_only', action='store_true', help="Calulate metrics without saving result")
    p.add('--use_cache', action='store_true', help="Use cached result when calculating metrics")
    p.add('--image_dir', type=str, help="Use other image directory for cached result")
    p.add('--image_offset', type=int, default=1, help="Image index offset for cached result")
    p.add('--use_train_mode', action='store_true', help="Use train mode when evaluating model")
    p.add('--save_mesh', action='store_true', help="Save posed mesh file")
    p.add('--save_rgb_with_alpha', action='store_true', help="Save rgb with alpha channel")
    p.add('--save_rgb_only', action='store_true', help="Only save rgb withour extra results")
    p.add('--output_dir_name', type=str, help="Use the specific output dir name instead")

    # Params fitting
    p.add('--fit_epochs', type=int, default=25, help="Number of fitting iterations per image")
    p.add('--fit_img_res', type=int, nargs=2, default=None, metavar=('width', 'height'))
    p.add('--fit_lr', type=float, default=1e-3, help="Learning rate for inputs")
    p.add('--fit_num_rays', type=int, default=-1, 
          help="Number of sampled rays to fit. Used for phase-1 model to lower memory usage.")
    p.add('--optim_class_input', default='torch.optim.Adam')
    p.add('--optim_args_input', type=yaml.safe_load, default={})
    p.add('--scheduler_class_input', default='torch.optim.lr_scheduler.StepLR')
    p.add('--scheduler_args_input', type=yaml.safe_load, default={'step_size': 10, 'gamma': 0.5})
    p.add('--optimize_latent', action='store_true', help="Optimize per-frame latent codes")
    p.add('--optimize_expression', action='store_true', help="Optimize expression inputs")
    p.add('--optimize_pose', action='store_true', help="Optimize pose inputs")
    p.add('--optimize_camera', action='store_true', help="Optimize camera inputs")

    # Metrics
    p.add('--metric_class', default='model.metrics.TestMetric')
    p.add('--metric_args', type=yaml.safe_load, default={}, help="Metric arguments")

    # Facial Reenactment
    p.add('--reenact_data_dir', help="Reenactment dataset directory path")
    p.add('--reenact_subdirs', nargs='+', help="Reenactment subdirectories")
    p.add('--reenact_subsample', type=yaml.safe_load, default=1, help="Subsample frames")
    p.add('--reenact_delta_transfer', action='store_true', help="Transfer delta to mean exp")

    # Expression editing
    p.add('--expression_offset', type=yaml.safe_load)
    p.add('--pose_offset', type=yaml.safe_load)


def add_mesh_export_arguments(p: configargparse.ArgParser) -> None:
    p.add('--export_iteration', type=int, help="Iteration to load weights from (default as latest)")
    p.add('--export_type',
          default='marching_cube',
          choices=['flame_template', 'flame_fitting', 'marching_cube', 'point_cloud'])

    # FLAME template mesh settings
    p.add('--flame_num_subdivision',
          type=int,
          default=0,
          help="Iterations to subdivide FLAME template mesh")
    p.add('--flame_sample_points', type=int, default=200000)
    p.add('--flame_fitting_loss_class', default='model.loss.MeshFittingLoss')
    p.add('--flame_fitting_loss_args',
          type=yaml.safe_load,
          default={},
          help="Fitting loss arguments")

    # Marching cube settings
    p.add('--mc_res_init', type=int, default=16, help="Initial resolution of marching cube")
    p.add('--mc_res_up', type=int, default=5, help="Resolution upscaling factor of marching cube")
    p.add('--mc_query_scale', type=float, default=1.2)
    p.add('--mc_center_offset_x', type=float, default=0.0)
    p.add('--mc_center_offset_y', type=float, default=0.1)
    p.add('--mc_center_offset_z', type=float, default=0.0)

    # Mesh simplification settings
    p.add('--mesh_angle_limit',
          type=float,
          default=120,
          help="Faces with normal at higher angle are simplified")
    p.add('--mesh_delete_high_angle', action='store_true', help="Delete faces at higher angle")
    p.add('--mesh_high_angle_face_percentage',
          type=float,
          default=0.15,
          help="Percentage of faces at higher angle to keep")
    p.add('--mesh_high_angle_expand_iterations',
          type=int,
          default=3,
          help="Also select neighbor faces to the higher angle faces")
    p.add('--mesh_target_face_count',
          type=int,
          default=10000,
          help="Target number of faces in the mesh")
    p.add('--mesh_remesh_iteration', type=int, default=0, help="Number of remesh iterations")
    p.add('--mesh_remesh_target_len_percentage',
          type=float,
          default=1,
          help="Target length for the remeshed mesh edges")

    # Optimizer & Scheduler
    p.add('--optim_class', default='torch.optim.AdamW')
    p.add('--optim_args', type=yaml.safe_load, default={})
    p.add('--scheduler_class', default='torch.optim.lr_scheduler.StepLR')
    p.add('--scheduler_args', type=yaml.safe_load, default={'step_size': 4000, 'gamma': 0.5})

    # Fitting hyperparameters
    p.add('--fitting_iterations', type=int, default=10000, help="Total number of iterations")
    p.add('--fitting_lr', type=float, default=1e-3, help="Learning rate for fitting optimization")

    # Logging
    p.add('--show_it', type=int, default=10, help="Num iterations to show progress")
    p.add('--save_it', type=int, default=1000, help="Num iterations to save snapshot")


def add_texture_export_arguments(p: configargparse.ArgParser) -> None:
    p.add('--export_iteration', type=int, help="Iteration to load weights from (default as latest)")
    p.add('--mesh_data_path', required=True, help="The path of mesh data json file")
    p.add('--resolution', type=int, default=1024, help="Resolution of exported textures")
    p.add('--supersampling', type=int, default=4, help="Level of supersampling in each texel")


def add_fine_tuning_arguments(p: configargparse.ArgParser) -> None:
    p.add('--finetune_model_class', default='model.rasterization.RasterizationModel')
    p.add('--finetune_model_args', type=yaml.safe_load, default={}, help="Model arguments")
    p.add('--mesh_data_path', required=True, help="The path of mesh data json file")

    # Dataloader
    p.add('--batch_size', type=int, default=2, help="Number of samples per batch")
    p.add('--no_shuffle', action='store_true', help="Do not shuffle train dataset")

    # Loss
    p.add('--finetune_loss_class', default='model.loss.FineTuningLoss')
    p.add('--finetune_loss_args', type=yaml.safe_load, default={}, help="Loss arguments")

    # Optimizer
    p.add('--optim_class', default='torch.optim.AdamW')
    p.add('--optim_class_input', default='torch.optim.Adam')
    p.add('--optim_args', type=yaml.safe_load, default={})
    p.add('--optim_args_input', type=yaml.safe_load, default={})
    p.add('--optimize_expression', action='store_true', help="Optimize expression inputs")
    p.add('--optimize_pose', action='store_true', help="Optimize pose inputs")
    p.add('--optimize_camera', action='store_true', help="Optimize camera inputs")

    # Scheduler
    p.add('--scheduler_class', default='torch.optim.lr_scheduler.ConstantLR')
    p.add('--scheduler_class_input', default='torch.optim.lr_scheduler.ConstantLR')
    p.add('--scheduler_args', type=yaml.safe_load, default={})
    p.add('--scheduler_args_input', type=yaml.safe_load, default={})

    # Training hyperparameters
    p.add('--iterations', type=int, default=1000000, help="Total number of iterations")
    p.add('--lr', type=float, default=1e-3, help="Learning rate for model params")
    p.add('--lr_input', type=float, default=1e-4, help="Learning rate for inputs")
    p.add('--weight_decay', type=float, default=0, help="Weight decay for model params")
    p.add('--clip_grad_norm', type=float, help="Gradient clipping max norm")

    # Logging
    p.add('--log_it', type=int, default=10, help="Num iterations to log")
    p.add('--show_it', type=int, default=100, help="Num iterations to display")
    p.add('--save_it', type=int, default=5000, help="Num iterations to save snapshot")
    p.add('--eval_it', type=int, default=1000, help="Num iterations to evaluate result")
    p.add('--export_it', type=int, default=5000, help="Num iterations to export mesh data")
    p.add('--avg_loss_it', type=int, default=100, help="Num iterations for average loss")


def make_run_args(p: configargparse.ArgParser, rundir_should_exist=False) -> dict:
    args, _ = p.parse_known_args()  # parse args
    args = dict(vars(args))  # convert to dict

    rundir = os.path.join(args['rundir'], args['run_name'])
    if rundir_should_exist:
        assert os.path.exists(rundir), f"Run directory {rundir} does not exist"
    else:
        ensure_dir(rundir)  # make run directory

    # write run config
    run_cfg_filename = os.path.join(rundir, "run_config.yaml")
    config = args.pop('config')
    args.pop('task')
    if config is None or os.path.abspath(config) != os.path.abspath(run_cfg_filename):
        p.print_values()  # print out values
        p.write_config_file(args, [run_cfg_filename])
    print('-' * 60)

    args.pop('rundir')
    args.pop('run_name')
    args['rundir'] = rundir
    return EasyDict(args)


if __name__ == "__main__":
    parser = build_base_parser()
    args, _ = parser.parse_known_args()  # parse args

    if args.task == 'train':
        from scripts.training import training_loop

        add_train_arguments(parser)
        args = make_run_args(parser)

        training_loop(**args)

    elif args.task == 'test':
        from scripts.testing import testing_loop

        add_test_arguments(parser)
        args = make_run_args(parser, rundir_should_exist=True)

        testing_loop(**args)

    elif args.task == 'mesh_export':
        from scripts.mesh_export import mesh_export

        add_mesh_export_arguments(parser)
        args = make_run_args(parser, rundir_should_exist=True)

        mesh_export(**args)

    elif args.task == 'texture_export':
        from scripts.texture_export import texture_export

        add_texture_export_arguments(parser)
        args = make_run_args(parser, rundir_should_exist=True)

        texture_export(**args)

    elif args.task == 'fine_tuning':
        from scripts.fine_tuning import fine_tuning_loop

        add_fine_tuning_arguments(parser)
        args = make_run_args(parser, rundir_should_exist=False)

        fine_tuning_loop(**args)

    else:
        assert 0, "Unknown task"
