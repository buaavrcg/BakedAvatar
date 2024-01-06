import numpy as np
import torch
import os
import matplotlib.pyplot as plt

from utils.misc_util import ensure_dir
from utils.image_util import visualize_images
from utils.render_util import alpha_integration


def visualize_outputs(output_dir,
                      inputs,
                      outputs,
                      targets,
                      iter=None,
                      near_depth=None,
                      far_depth=None):
    for batch_idx in range(inputs['img_res'].shape[0]):
        if iter is not None:
            dir = os.path.join(output_dir, inputs['sub_dir'][batch_idx], f"iter_{iter}")
        else:
            dir = os.path.join(output_dir, inputs['sub_dir'][batch_idx], "rendering")
        ensure_dir(dir, False)

        def make_filename(type):
            return os.path.join(dir, f"{type}_{inputs['sub_id'][batch_idx].item() + 1}.png")

        img_res = inputs['img_res'][batch_idx].cpu()
        rgb_target = targets['rgb'][None, batch_idx]
        rgb_output = outputs['rgb'][None, batch_idx]
        mask_target = targets['mask'][None, batch_idx].unsqueeze(-1).float()
        mask_output = outputs['mask'][None, batch_idx].unsqueeze(-1).float()

        has_flame_weights = outputs['shapedirs'] is not None
        if has_flame_weights:
            shapedirs, lbs_weights = visualize_flame_weights(outputs, batch_idx)
            (normal_output, lbs_weights_output, shapedirs_output), depth, _ = alpha_integration(
                (
                    outputs['normal'][None, batch_idx].detach(),
                    lbs_weights[None],
                    shapedirs[None],
                ),
                outputs['dist'][None, batch_idx].detach(),
                outputs['alpha_manifold'][None, batch_idx].detach(),
                outputs['mask_manifold'][None, batch_idx].detach(),
                depth_background=torch.max(outputs['dist'][None, batch_idx].detach()),
            )
        else:
            (normal_output, ), depth, _ = alpha_integration(
                (outputs['normal'][None, batch_idx].detach(), ),
                outputs['dist'][None, batch_idx].detach(),
                outputs['alpha_manifold'][None, batch_idx].detach(),
                outputs['mask_manifold'][None, batch_idx].detach(),
                depth_background=torch.max(outputs['dist'][None, batch_idx].detach()),
            )

        normal_output = torch.nn.functional.normalize(normal_output, dim=-1)
        normal_output = (normal_output + 1) / 2  # (1, num_rays, 3)
        if near_depth is None or far_depth is None:
            near = torch.min(depth)
            far = torch.max(depth)
        else:
            near = near_depth
            far = far_depth
        depth_output = torch.clamp((far - depth.unsqueeze(-1)) / (far - near), 0, 1)

        if has_flame_weights:
            visualize_images(img_res,
                             rgb_target,
                             rgb_output,
                             mask_target,
                             mask_output,
                             normal_output,
                             depth_output,
                             lbs_weights_output,
                             shapedirs_output,
                             filename=make_filename("result"),
                             nrow=4)
        else:
            visualize_images(img_res,
                             rgb_target,
                             rgb_output,
                             mask_target,
                             mask_output,
                             normal_output,
                             depth_output,
                             filename=make_filename("result"),
                             nrow=4)

        alpha = outputs['alpha_manifold'][batch_idx].swapaxes(0, 1).unsqueeze(-1)
        mask = outputs['mask_manifold'][batch_idx].swapaxes(0, 1).unsqueeze(-1).float()
        feat = outputs['feature_manifold'][batch_idx].swapaxes(0, 1)  # (num_levels, num_rays, d_f)
        feat_rgb = feat[:, :, :3] * alpha \
                 + inputs['background_rgb'][batch_idx][None, None, :] * (1 - alpha)
        dist = outputs['dist'][batch_idx].swapaxes(0, 1).unsqueeze(-1)  # (num_levels, num_rays, 1)
        if near_depth is None or far_depth is None:
            near = torch.min(dist)
            far = torch.max(dist)
        else:
            near = near_depth
            far = far_depth
        dist = torch.clamp((far - dist) / (far - near), 0, 1)
        normal = outputs['normal'][batch_idx].swapaxes(0, 1)  # (num_levels, num_rays, 3)
        normal = (normal.detach() + 1) / 2

        if has_flame_weights:
            visualize_images(img_res,
                             feat_rgb,
                             dist,
                             mask,
                             normal,
                             alpha,
                             visualize_texture_coef(outputs, batch_idx).swapaxes(0, 1),
                             lbs_weights.swapaxes(0, 1),
                             shapedirs.swapaxes(0, 1),
                             filename=make_filename("manifold"))
        else:
            visualize_images(img_res,
                             feat_rgb,
                             dist,
                             mask,
                             normal,
                             alpha,
                             visualize_texture_coef(outputs, batch_idx).swapaxes(0, 1),
                             filename=make_filename("manifold"))


def visualize_flame_weights(outputs, batch_idx):
    cmap = plt.get_cmap('Set1')
    identity_red = cmap.colors[0]
    global_green = cmap.colors[2]
    neck_blue = cmap.colors[1]
    white = (1.0, 1.0, 1.0)

    mask_manifold = outputs['mask_manifold'][batch_idx]
    lbs_weights = outputs['lbs_weights'][batch_idx].detach()  # (num_rays, num_levels, 5)
    shapedirs = outputs['shapedirs'][batch_idx][..., 0].detach()  # (num_rays, num_levels, 3)
    shapedirs = torch.clamp((shapedirs * 50 + 1) / 2, 0, 1)

    if lbs_weights.shape[-1] == 5:
        colors = np.array([global_green, neck_blue, white, white, white])
    elif lbs_weights.shape[-1] == 6:
        colors = np.array([identity_red, global_green, neck_blue, white, white, white])
    else:
        assert 0, f"Unsupported num of bones {lbs_weights.shape[-1]}."
    colors = torch.from_numpy(colors).to(lbs_weights.device)  # (5, 3)
    lbs_weights = torch.sum(colors[None, None] * lbs_weights[..., None], -2)

    mask_outside = torch.logical_not(mask_manifold)
    shapedirs[mask_outside] = 0.0
    lbs_weights[mask_outside] = 0.0

    return shapedirs, lbs_weights


def visualize_texture_coef(outputs, batch_idx):
    cmap = plt.get_cmap('tab10')

    # texture_coef: # (num_rays, num_levels, num_texture_basis)
    texture_coef = outputs['texture_coef'][batch_idx].detach()
    colors = np.array([cmap.colors[i % 10] for i in range(texture_coef.shape[-1])])
    colors = torch.from_numpy(colors).to(texture_coef.device)
    texture_coef = torch.sum(colors[None, None] * texture_coef[..., None], -2)

    mask_manifold = outputs['mask_manifold'][batch_idx]
    mask_outside = torch.logical_not(mask_manifold)
    texture_coef[mask_outside] = 0.0

    return texture_coef