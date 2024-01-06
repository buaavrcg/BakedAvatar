import numpy as np
import torch
import cv2


def load_K_Rt_from_P(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Args:
        P: Projection matrix of shape (3, 4)
    Returns:
        K: Camera intrinsics of shape (3, 3)
        Rt: Camera extrinsics of shape (4, 4)

    Code based on https://github.com/zhengyuf/IMavatar
    """
    out = cv2.decomposeProjectionMatrix(P)
    K, R, t = out[:3]
    K = K / K[2, 2]
    Rt = np.eye(4, dtype=np.float32)
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = (t[:3] / t[3])[:, 0]
    return K, Rt


def get_projection_and_view(K: torch.Tensor,
                            Rt: torch.Tensor,
                            img_res: torch.Tensor,
                            near: float = 1.0,
                            far: float = 10.0) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        K: Camera intrinsics of shape (batch_size, 3, 3)
        Rt: Camera extrinsics of shape (batch_size, 4, 4)
        img_res: Image resolution of shape (batch_size, 2)
        near: float. Near plane.
        far: float. Far plane.
    Returns:
        P: Projection matrix of shape (batch_size, 3, 4)
        V: View matrix of shape (batch_size, 4, 4)
    """
    assert K.ndim == 3 and K.shape[1] == 3 and K.shape[2] == 3
    assert Rt.ndim == 3 and Rt.shape[1] == 4 and Rt.shape[2] == 4
    assert img_res.ndim == 2 and img_res.shape[1] == 2

    fx, fy = K[:, 0, 0], K[:, 1, 1]
    cx, cy = K[:, 0, 2], K[:, 1, 2]
    sk = K[:, 0, 1].unsqueeze(-1)
    assert torch.all(sk == 0.0), "Non-zero skew is not supported."
    W_half, H_half = img_res[:, 0].float() / 2, img_res[:, 1].float() / 2
    zn, zf = near, far
    _0 = torch.zeros_like(fx)
    _1 = torch.ones_like(fx)

    proj = torch.stack([
        torch.stack([-fx / W_half, _0, 1.0 - cx / W_half, _0], -1),
        torch.stack([_0, -fy / H_half, 1.0 - cy / H_half, _0], -1),
        torch.stack([_0, _0, -(zf + zn) / (zf - zn) * _1, -2 * zf * zn / (zf - zn) * _1], -1),
        torch.stack([_0, _0, -_1, _0], -1),
    ], -2)
    view = torch.inverse(Rt)

    return proj, view


def uv_to_rays(uv: torch.Tensor, K: torch.Tensor,
               Rt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert uv coordinates to rays in world space.

    Args:
        uv: (batch_size, num_rays, 2). UV coordinates.
        K: (batch_size, 3, 3). Camera intrinsics.
        Rt: (batch_size, 4, 4). Camera extrinsics in matrix form.
    Returns:
        rays_o: (batch_size, 3). The origin of rays in world space.
        rays_d: (batch_size, num_rays, 3). The direction of rays in world space.
    """
    assert K.ndim == 3 and K.shape[1] == 3 and K.shape[2] == 3
    assert Rt.ndim == 3 and Rt.shape[1] == 4 and Rt.shape[2] == 4

    rays_o = Rt[:, :3, 3]  # (batch_size, 3)
    fx, fy = K[:, 0, 0].unsqueeze(-1), K[:, 1, 1].unsqueeze(-1)
    cx, cy = K[:, 0, 2].unsqueeze(-1), K[:, 1, 2].unsqueeze(-1)
    sk = K[:, 0, 1].unsqueeze(-1)

    x = uv[:, :, 0] - cx
    y = uv[:, :, 1] - cy
    pixel_coords = torch.stack([
        -1 * (x - sk * y / fy) / fx,
        -1 * y / fy,
        -1 * torch.ones_like(x),
        torch.ones_like(x),
    ], -1)
    world_coords = torch.bmm(Rt, pixel_coords.permute(0, 2, 1)).permute(0, 2, 1)[:, :, :3]

    rays_d = world_coords - rays_o.unsqueeze(1)  # (batch_size, num_rays, 3)
    rays_d = torch.nn.functional.normalize(rays_d, dim=-1)

    return rays_o, rays_d


def ray_sphere_intersection(rays_o: torch.Tensor, rays_d: torch.Tensor,
                            radius: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Intersect rays with the sphere at origin.

    Args:
        rays_o: (*, 3). The origin of rays.
        rays_d: (*, 3). The direction of rays.
        radius: float. The radius of sphere at origin.
    Returns:
        d_intersect: (*, 2). The near and far depth of intersection points.
        mask_intersect: (*,). Whether this ray has intersections with sphere.
    """
    A = torch.sum(rays_d * rays_d, dim=-1)
    B = 2 * torch.sum(rays_o * rays_d, dim=-1)
    C = torch.sum(rays_o * rays_o, dim=-1) - radius**2

    delta = B * B - 4 * A * C
    mask_intersect = delta > 1e-9

    sqrt_delta = torch.zeros_like(delta)
    sqrt_delta[mask_intersect] = torch.sqrt(delta[mask_intersect])
    inv_2A = 1.0 / (2 * A)
    d_near = (-B - sqrt_delta) * inv_2A
    d_far = (-B + sqrt_delta) * inv_2A
    d_intersect = torch.stack([d_near, d_far], dim=-1)

    return d_intersect, mask_intersect


def perturb_depth_samples(d_samples: torch.Tensor) -> torch.Tensor:
    """
    Randomly perturb depth samples from linearly sampled samples.

    Args:
        d_samples: (*, num_samples). Uniform sampled depth samples.
    Returns:
        d_samples: (*, num_samples). Perturbed depth samples.
    """
    # get intervals between samples
    mids = .5 * (d_samples[..., 1:] + d_samples[..., :-1])
    upper = torch.cat([mids, d_samples[..., -1:]], -1)
    lower = torch.cat([d_samples[..., :1], mids], -1)

    # stratified samples in those intervals
    t_rand = torch.rand_like(d_samples)
    t_rand[..., -1:] = 1.0
    d_samples = lower + (upper - lower) * t_rand

    return d_samples


def alpha_integration(values: tuple[torch.Tensor],
                      depth: torch.Tensor,
                      alpha: torch.Tensor,
                      mask: torch.Tensor = None,
                      depth_background: float = 1e2):
    """
    Combine values along rays from near to far with (masked) alpha compositing.
    Args:
        values: tuple of (*, num_samples, D). Values to be combined.
        depth: (*, num_samples). Depth values of each sample.
        alpha: (*, num_samples). Alpha values at each samples.
        mask: (*, num_samples). Whether this sample is valid. If none, all samples are valid.
    Returns:
        *values: tuple of (*, D). Weighted sum of values along the rays.
        depth: (*,). Weighted depth along the rays.
        weight_background: (*,). Weight of background.
    """
    if mask is not None:
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))  # (*, num_samples)

    alpha_shifted = torch.cat([torch.ones_like(alpha[..., :1]), 1 + 1e-10 - alpha], dim=-1)
    transmitteance = torch.cumprod(alpha_shifted, dim=-1)[..., :-1]  # (*, num_samples)
    weights = alpha * transmitteance  # (*, num_samples)
    weights_sum = torch.sum(weights, dim=-1)  # (*,)

    weight_background = torch.clamp(1 - weights_sum, min=0)  # (*,)
    values = [torch.sum(value * weights[..., None], dim=-2) for value in values]  # (*, D)
    depth = torch.sum(depth * weights, dim=-1) + depth_background * weight_background  # (*,)

    return values, depth, weight_background


def density_integration(values: tuple[torch.Tensor],
                        depth: torch.Tensor,
                        sigma: torch.Tensor,
                        mask: torch.Tensor = None,
                        depth_background: float = 1e2,
                        delta_alpha=1.0):
    """
    Combine values along rays from near to far with (masked) density compositing.
    Args:
        values: tuple of (*, num_samples, D). Values to be combined.
        depth: (*, num_samples). Depth values of each sample.
        sigma: (*, num_samples). Density values in [0, +inf) at each samples.
        mask: (*, num_samples). Whether this sample is valid. If none, all samples are valid.
    Returns:
        *values: tuple of (*, D). Weighted sum of values along the rays.
        depth: (*,). Weighted depth along the rays.
        weight_background: (*,). Weight of background.
    """
    delta = (depth[..., 1:] - depth[..., :-1]) / 2  # (*, num_samples - 1)
    delta = torch.cat([
        delta[..., :1] * 2,
        delta[..., :-1] + delta[..., 1:],
        delta[..., -1:] * 2,
    ], -1)
    alpha = 1 - torch.exp(-delta_alpha * delta * sigma)  # (*, num_samples)
    if mask is not None:
        alpha = torch.where(mask, alpha, torch.zeros_like(alpha))  # (*, num_samples)

    alpha_shifted = torch.cat([torch.ones_like(alpha[..., :1]), 1 + 1e-10 - alpha], dim=-1)
    transmitteance = torch.cumprod(alpha_shifted, dim=-1)[..., :-1]  # (*, num_samples)
    weights = alpha * transmitteance  # (*, num_samples)
    weights_sum = torch.sum(weights, dim=-1)  # (*,)

    weight_background = torch.clamp(1 - weights_sum, min=0)  # (*,)
    values = [torch.sum(value * weights[..., None], dim=-2) for value in values]  # (*, D)
    depth = torch.sum(depth * weights, dim=-1) + depth_background * weight_background  # (*,)

    return values, depth, weight_background