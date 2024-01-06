import numpy as np
import torch
import tqdm
import os
from imageio import imwrite
from accelerate import Accelerator

from utils.training_util import seed_everything
from utils.misc_util import ensure_dir, find_latest_model_path, construct_class_by_name, Logger


def sample_texel(obj_path, resolution):
    """
    Generate texel sampling points on the uv plane and their correspondence
    spatial points on the mesh surface.
    Args:
        obj_path: (str) path to the mesh obj file
        resolution: (float) The resolution of texture sampling
    Returns:
        sampled_points: float32 (M, 3) sampled points on the mesh surface
        sampled_pixel_uvs: int32 (M, 2) pixel UV coordinates of sampled surface points
    """
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(obj_path)
    ms.generate_sampling_texel(texturew=resolution, textureh=resolution)
    ms.set_current_mesh(0)
    ms.generate_sampling_texel(texturew=resolution, textureh=resolution, texturespace=True)

    sampled_points = ms.mesh(1).vertex_matrix().astype(np.float32)
    sampled_pixel_uvs = ms.mesh(2).vertex_matrix()
    sampled_pixel_uvs = sampled_pixel_uvs[:, :2].astype(np.int32)

    return sampled_points, sampled_pixel_uvs


def texture_export(
    rundir,
    seed,
    use_cpu,
    # Dataset (only for loading shape params)
    dataset_class,
    data_dir,
    train_subdirs,
    img_res,
    train_subsample,
    # Model
    model_class,
    model_args,
    # Exporting
    export_iteration,
    resolution,
    supersampling,
    mesh_data_path,
    **kwargs,
):
    accel = Accelerator(cpu=use_cpu)
    seed_everything(seed + accel.process_index)  # set seed
    checkpoints_dir = os.path.join(rundir, "checkpoints")
    if accel.is_local_main_process:
        assert os.path.exists(checkpoints_dir), "No checkpoints found!"
        assert os.path.exists(mesh_data_path), "No mesh data found!"
        Logger(os.path.join(rundir, "exporting_log.txt"), "w+")

    # Load dataset
    train_dataset = construct_class_by_name(class_name=dataset_class,
                                            data_dir=data_dir,
                                            sub_dirs=train_subdirs,
                                            img_res=img_res,
                                            num_rays=-1,
                                            subsample=train_subsample,
                                            use_semantics=False,
                                            no_gt=False)
    accel.print(f"Loaded {len(train_dataset)} testing frames from {data_dir}/{train_subdirs}.")

    # Build and load model
    model = construct_class_by_name(class_name=model_class,
                                    shape_params=train_dataset.get_shape_params(),
                                    canonical_exp=train_dataset.get_mean_expression(),
                                    **model_args)
    if export_iteration is not None:
        load_ckpt_dir = os.path.join(checkpoints_dir, f"iter_{export_iteration}")
    else:
        load_ckpt_dir = find_latest_model_path(checkpoints_dir)
    model_state = torch.load(os.path.join(load_ckpt_dir, "model.pth"), accel.device)
    model.load_state_dict(model_state['model'], strict=True)
    it, sample_count = model_state['it'], model_state['sample_count']
    accel.print(f'Loaded checkpoint (iter {it}, samples {sample_count}) from: {load_ckpt_dir}')
    model = accel.prepare(model)

    # Load mesh data
    mesh_data = torch.load(mesh_data_path)
    mesh_data_dir = os.path.dirname(mesh_data_path)
    output_dir = os.path.join(mesh_data_dir, "tex")
    ensure_dir(output_dir)

    # start exporting textures of the mesh
    accel.print(f"Start exporting texture...")
    model.eval()
    batch_size = 200000

    for level_idx, data in enumerate(mesh_data['meshes']):
        obj_path = os.path.join(mesh_data_dir, "obj", f"mesh_{level_idx}_canonical.obj")

        # sample texel points on the uv plane
        sampled_points, sampled_uvs = sample_texel(obj_path, resolution * supersampling)

        # create supersampled texture planes (N_basis, H, W, C)
        position_texture = torch.zeros(resolution,
                                       resolution,
                                       model.dim_pos_feature,
                                       device=accel.device)
        radiance_textures = torch.zeros(model.num_texture_basis,
                                        resolution,
                                        resolution,
                                        model.dim_radiance_feature + 1,
                                        device=accel.device)
        num_samples = torch.zeros(resolution, resolution, device=accel.device)

        # evaluate radiance net for each batch of points
        for i in tqdm.tqdm(range(0, len(sampled_points), batch_size),
                           desc=f"Evaluating radiance net for level {level_idx}"):
            points = torch.from_numpy(sampled_points[i:i + batch_size]).to(accel.device)
            uvs = torch.from_numpy(sampled_uvs[i:i + batch_size]).to(accel.device)
            uvs = torch.floor_divide(uvs, supersampling).long()

            with torch.no_grad():
                pos_feature, features, alpha = model.radiance_net.get_static_features(points)

            pos_feature = pos_feature.view(len(points), -1)
            features = features.view(len(points), model.num_texture_basis, -1)
            alpha = alpha.reshape(len(points), 1, 1).repeat(1, model.num_texture_basis, 1)
            radiance_feature = torch.cat([features, alpha], dim=-1)

            # accumulate features to the texture planes
            position_texture[uvs[:, 1], uvs[:, 0]] += pos_feature
            radiance_textures[:, uvs[:, 1], uvs[:, 0]] += radiance_feature.permute(1, 0, 2)
            num_samples[uvs[:, 1], uvs[:, 0]] += 1

        # divide the number of samples per texel
        num_samples[num_samples == 0] = 1  # avoid dividing by zero
        position_texture /= num_samples[:, :, None]
        radiance_textures /= num_samples[None, :, :, None]

        # save textures
        assert torch.all(0 <= position_texture) and torch.all(position_texture <= 1.0)
        assert torch.all(0 <= radiance_textures) and torch.all(radiance_textures <= 1.0)
        position_texture = np.clip(position_texture.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        radiance_textures = np.clip(radiance_textures.cpu().numpy() * 255, 0, 255).astype(np.uint8)
        data['position_texture'] = position_texture
        data['radiance_textures'] = radiance_textures

        imwrite(
            os.path.join(output_dir, f"mesh_{level_idx}_pos_feature.png"),
            np.stack([
                np.mean(position_texture, axis=-1),
                np.min(position_texture, axis=-1),
                np.max(position_texture, axis=-1),
            ],
                     axis=-1).astype(np.uint8))
        for i, t in enumerate(radiance_textures):
            t = np.concatenate((t[:, :, :3], t[:, :, -1:]), axis=-1)
            imwrite(os.path.join(output_dir, f"mesh_{level_idx}_radiance_basis{i}.png"), t)

    mesh_data['dim_pos_feature'] = model.dim_pos_feature
    mesh_data['num_texture_basis'] = model.num_texture_basis
    mesh_data['spatial_mlp_temperature'] = model.radiance_net.tex_projector.temperature
    accel.print(f"Finished exporting textures, start exporting MLP weights.")

    # export shading MLP weights
    weights_dict = model.radiance_net.tex_projector.export_weights()
    for k, weights in weights_dict.items():
        accel.print(f"{k}: {weights.shape} {weights}")
        weights_dict[k] = weights.cpu().numpy().astype(np.float32)
    mesh_data['global_mlp'] = weights_dict

    accel.print(f"Finished exporting weights, result saved to {mesh_data_path}.")
    torch.save(mesh_data, mesh_data_path)
