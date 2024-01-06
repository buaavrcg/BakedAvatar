import math
import torch
import numpy as np

from utils.render_util import uv_to_rays, ray_sphere_intersection, perturb_depth_samples


class ManifoldRenderer(torch.nn.Module):
    def __init__(self,
                 num_manifold_levels,
                 manifold_level_start,
                 manifold_level_end,
                 manifold_samples=64,
                 manifold_secant_steps=0,
                 bounding_sphere_radius=1.0,
                 near_distance=0,
                 far_distance=1e2,
                 perturb_when_training=False,
                 use_gt_mask_when_training=False,
                 use_presort=False):
        super().__init__()
        self.manifold_levels = torch.nn.Parameter(
            torch.linspace(manifold_level_start, manifold_level_end, num_manifold_levels),
            requires_grad=False,
        )
        self.manifold_samples = manifold_samples
        self.manifold_secant_steps = manifold_secant_steps
        self.bounding_sphere_radius = bounding_sphere_radius
        self.near_distance = near_distance
        self.far_distance = far_distance
        self.perturb_when_training = perturb_when_training
        self.use_gt_mask_when_training = use_gt_mask_when_training
        self.use_presort = use_presort

    def find_closest_interval(self, depths, scalars, levels, mask_valid_scalar, mask_gt=None):
        """
        Find the closest descending scalar intervals that contains the given levels.
        Args:
            depths: (N, num_samples). The depths at sample points on rays.
            scalars: (N, num_samples). The scalars at sample points on rays.
            levels: (num_levels,). The manifold levels.
            mask_valid_scalar: (N, num_samples). The mask indicate if scalars are valid.
            mask_gt: (N,). The ground truth mask of the object.
        Returns:
            d_front: (N, num_levels). Front endpoint depth of the interval.
            d_back: (N, num_levels). Back endpoint depth of the interval.
            s_front: (N, num_levels). Front endpoint scalar of the interval.
            s_back: (N, num_levels). Back endpoint scalar of the interval.
        """
        num_samples = scalars.shape[1]
        num_levels = levels.shape[0]
        d_front = depths[:, None, :-1].expand(-1, num_levels, -1)  # (N, num_levels, num_samples-1)
        d_back = depths[:, None, 1:].expand(-1, num_levels, -1)  # (N, num_levels, num_samples-1)
        s_front = scalars[:, None, :-1].expand(-1, num_levels, -1)  # (N, num_levels, num_samples-1)
        s_back = scalars[:, None, 1:].expand(-1, num_levels, -1)  # (N, num_levels, num_samples-1)
        l = levels[None, :, None]  # (1, num_levels, 1)
        m_front = mask_valid_scalar[:, None, :-1]  # (N, 1, num_samples-1)
        m_back = mask_valid_scalar[:, None, 1:]  # (N, 1, num_samples-1)
        m_valid = m_front & m_back  # (N, 1, num_samples-1)

        # 1. find the closest descending interval that contains the given levels.
        # 2. if there is no match, choose the descending interval with smallest scalar.
        in_interval = (s_front >= l) & (l >= s_back) & m_valid  # (N, num_levels, num_samples-1)
        r_cloest = torch.sign(l - s_back) * torch.arange(num_samples - 1, 0, -1, device=l.device)
        ind_closest = torch.argmax(r_cloest, dim=2, keepdim=True)  # (N, num_levels, 1)
        ind_lowest = torch.argmin(s_back, dim=2, keepdim=True)  # (N, num_levels, 1)
        mask_surface = torch.any(in_interval, dim=2, keepdim=True)  # (N, num_levels, 1)
        if mask_gt is not None:
            mask_surface = mask_surface & mask_gt[:, None, None]
        indices = torch.where(mask_surface, ind_closest, ind_lowest)

        # gather the endpoints of the best intervals
        d_front = torch.gather(d_front, 2, indices).squeeze(2)  # (N, num_levels)
        d_back = torch.gather(d_back, 2, indices).squeeze(2)  # (N, num_levels)
        s_front = torch.gather(s_front, 2, indices).squeeze(2)  # (N, num_levels)
        s_back = torch.gather(s_back, 2, indices).squeeze(2)  # (N, num_levels)

        return d_front, d_back, s_front, s_back

    def find_exact_intersection(
        self,
        d_front,
        d_back,
        s_front,
        s_back,
        levels,
        manifold_fn,
        rays_o,
        rays_d,
        rays_idx,
        mask_gt=None,
        eps=1e-4,
    ):
        """
        Find the exact intersection point of the manifold levels with the rays.
        If manifold_secant_steps > 0, secant method is used to refine intersection points.
        Args:
            d_front: (N, num_levels). Front endpoint depth of the interval.
            d_back: (N, num_levels). Back endpoint depth of the interval.
            s_front: (N, num_levels). Front endpoint scalar of the interval.
            s_back: (N, num_levels). Back endpoint scalar of the interval.
            levels: (num_levels,). The manifold levels.
            manifold_fn: callable, M(x_d, idx) -> (s, x_c, mask).
            rays_o: (N, 3). The origin of the rays.
            rays_d: (N, 3). The direction of the rays.
            rays_idx: long of (N,). The batch index of the rays.
            mask_gt: (N,). The ground truth mask of the object.
            eps: A small number to avoid division by zero.
        Returns:
            d_intersect: (N, num_levels). The exact intersection depth.
            mask_intersect: (N, num_levels). The mask indicating if the intersection exists.
        """
        l = levels[None, :]  # (1, num_levels)
        s_diff = s_front - s_back  # (N, num_levels)

        # whether the difference between the front and back scalar is large enough
        mask_valid_diff = torch.abs(s_diff) > eps  # (N, num_levels)

        # whether the manifold level is between the front and back scalar
        mask_intersect = (s_front >= l) & (l >= s_back)  # (N, num_levels)
        if mask_gt is not None:
            mask_intersect = mask_intersect & mask_gt[:, None]

        # using one-iteration secant method to find the intersection depth
        # if we can't find a valid intersection depth, use the back point of the interval
        mask_secant = mask_intersect & mask_valid_diff  # (N, num_levels)
        d_secant = ((l - s_back) * d_front + (s_front - l) * d_back) / s_diff
        d_intersect = torch.where(mask_secant, d_secant, d_back)  # (N, num_levels)

        if self.manifold_secant_steps > 0 and torch.any(mask_secant):
            # do secant method only for valid intersections found
            d_front_secant = d_front[mask_secant]  # (N_secant,)
            d_back_secant = d_back[mask_secant]  # (N_secant,)
            s_front_secant = s_front[mask_secant]  # (N_secant,)
            s_back_secant = s_back[mask_secant]  # (N_secant,)
            d_secant = d_intersect[mask_secant]  # (N_secant,)
            l_secant = l.expand(d_front.shape[0], -1)[mask_secant]  # (N_secant,)

            n_levels = levels.shape[0]
            rays_o = rays_o[:, None, :].expand(-1, n_levels, -1)[mask_secant]  # (N_secant, 3)
            rays_d = rays_d[:, None, :].expand(-1, n_levels, -1)[mask_secant]  # (N_secant, 3)
            rays_idx = rays_idx[:, None].expand(-1, n_levels)[mask_secant]  # (N_secant,)

            for _ in range(self.manifold_secant_steps):
                x_d_mid = rays_o + d_secant[:, None] * rays_d  # (N_secant, 3)
                s_mid, _, mask_valid_s = manifold_fn(x_d_mid, rays_idx)  # (N_secant,)

                ind_front = (s_mid > l_secant) & mask_valid_s
                ind_back = (s_mid < l_secant) & mask_valid_s
                if torch.any(ind_front):
                    d_front_secant[ind_front] = d_secant[ind_front]
                    s_front_secant[ind_front] = s_mid[ind_front]
                if torch.any(ind_back):
                    d_back_secant[ind_back] = d_secant[ind_back]
                    s_back_secant[ind_back] = s_mid[ind_back]

                s_diff_secant = s_front_secant - s_back_secant  # (N_secant,)
                d_secant = torch.where(
                    torch.abs(s_diff_secant) > eps,
                    ((l_secant - s_back_secant) * d_front_secant +
                     (s_front_secant - l_secant) * d_back_secant) / s_diff_secant,
                    d_secant)  # (N_secant,)

            # print('secant diff', torch.abs(d_intersect[mask_secant] - d_secant))

            # update more accurate intersections from secant method
            d_intersect[mask_secant] = d_secant  # (N, num_levels)

        return d_intersect, mask_intersect

    def ray_march_manifold(
        self,
        manifold_fn,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        rays_idx: torch.LongTensor,
        mask_gt: torch.Tensor,
        levels: torch.Tensor,
        num_samples: int,
        perturb=False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ray marching in deformed manifold field to find surface points in canonical space.
        The function returns the canonical surface points as well as d of the formula
            ray(d) = ray_o + d * ray_d

        Args:
            manifold_fn: callable, M(x_d, idx) -> (s, x_c, mask). The function takes deformed 
                points x_d and ray batch indices idx, returns manifold scalars s, canonical
                surface points x_c and a valid mask indicating if the found points are valid.
            rays_o: (batch_size, num_rays, 3). The origin of the rays.
            rays_d: (batch_size, num_rays, 3). The direction of the rays.
            rays_idx: long of (batch_size, num_rays). The batch index of the rays.
            mask_gt: bool of (batch_size, num_rays). The ground truth object mask of the rays.
            levels: (num_levels,). The levels of the manifold field.
            num_samples: int. The number of samples for each ray.
            perturb: bool. If True, each ray is sampled at stratified random points.
        Returns:
            dist: (batch_size, num_rays, num_levels). The d in formula ray(d) = ray_o + d * ray_d.
            mask: bool of (batch_size, num_rays, num_levels). Whether the intersects are valid.
        """
        device = rays_o.device

        # find the min/max distance range, bounding sphere mask of ray samples
        d_range, mask_bound = ray_sphere_intersection(rays_o, rays_d, self.bounding_sphere_radius)
        d_near = torch.clamp(d_range[..., 0], min=self.near_distance)  # (batch_size, num_rays)
        d_far = torch.clamp(d_range[..., 1], max=self.far_distance)  # (batch_size, num_rays)
        mask_bound = mask_bound & (d_near < d_far)  # (batch_size, num_rays)

        # generate sample distances on rays linearly
        t = torch.linspace(0, 1, num_samples, device=device)[None, :]  # (1, n_samples)
        d_samples = d_near[mask_bound][:, None] * (1 - t) \
                  + d_far[mask_bound][:, None] * t  # (n_mask_rays, n_samples)
        if perturb:
            d_samples = perturb_depth_samples(d_samples)

        # evaluate manifold field at the sample points in deformed space
        # x_d: (n_mask_rays, n_samples, 3)
        # idx: (n_mask_rays, n_samples)
        x_d = rays_o[mask_bound][:, None, :] + rays_d[mask_bound][:, None, :] * d_samples[..., None]
        idx = rays_idx[mask_bound][:, None].expand(-1, num_samples)

        scalars, x_c, mask_valid = manifold_fn(x_d.reshape(-1, 3), idx.reshape(-1))
        scalars = scalars.reshape_as(d_samples)  # (n_mask_rays, n_samples)
        mask_valid = mask_valid.reshape_as(d_samples)  # (n_mask_rays, n_samples)
        if self.use_gt_mask_when_training and mask_gt is not None and self.training:
            mask_gt_surface = mask_gt[mask_bound]
        else:
            mask_gt_surface = None

        # find the first valid intersection interval for each level
        d_front, d_back, s_front, s_back = self.find_closest_interval(d_samples, scalars, levels,
                                                                      mask_valid, mask_gt_surface)

        # calculate the exact intersection depth for each level
        d_intersect, mask_intersect = self.find_exact_intersection(
            d_front, d_back, s_front, s_back, levels, manifold_fn, rays_o[mask_bound],
            rays_d[mask_bound], rays_idx[mask_bound], mask_gt_surface)

        # sort intersections by distance
        if self.use_presort:
            d_intersect = torch.flip(d_intersect, [1])
            mask_intersect = torch.flip(mask_intersect, [1])
        else:
            _, indices = torch.sort(d_intersect, dim=-1)
            d_intersect = torch.gather(d_intersect, -1, indices)  # (n_mask_rays, num_levels)
            mask_intersect = torch.gather(mask_intersect, -1, indices)  # (n_mask_rays, num_levels)

        # merge results with rays outside the bounding sphere
        dist = torch.zeros(*rays_o.shape[:2], d_intersect.shape[1], device=device)
        mask = torch.zeros(dist.shape, dtype=torch.bool, device=device)
        dist[mask_bound] = d_intersect
        mask[mask_bound] = mask_intersect

        # for rays outside the bounding sphere, which are not sampled with manifold field, they
        # should not be part of the ground truth object mask, otherwise the bounding sphere is too
        # small for this object. Their depths are distance of the origin projected to ray direction.
        mask_outside = ~mask_bound
        if torch.any(mask_outside):
            assert mask_gt is None or torch.any(mask_outside & mask_gt), \
                'Some rays outside the bounding sphere are intersected with the ground truth ' \
                'object mask, which is not expected. Please increase the bounding sphere radius.'
            rays_o_out = rays_o[mask_outside]  # (n_outside_rays, 3)
            rays_d_out = rays_d[mask_outside]  # (n_outside_rays, 3)
            d_out = -torch.bmm(rays_d_out.view(-1, 1, 3), rays_o_out.view(-1, 3, 1)).squeeze()
            dist[mask_outside] = d_out

        return dist, mask

    def forward(
        self,
        manifold_fn,
        uv,
        intrinsic,
        extrinsic,
        mask=None,
        **extra_input,
    ):
        """
        Find the intersection points of rays with the manifold field.
        Args:
            manifold_fn: callable, M(x_d, idx) -> (s, x_c, mask). The function takes deformed 
                points x_d and ray batch indices idx, returns manifold scalars s, canonical
                surface points x_c and a valid mask indicating if the found points are valid.
            uv: (batch_size, num_rays, 2). The uv coordinates of the rays.
            intrinsic: (batch_size, 3, 3). The intrinsic matrix of the camera.
            extrinsic: (batch_size, 4, 4). The extrinsic matrix of the camera.
            mask: bool of (batch_size, num_rays). The ground truth object mask of the rays.
        Returns:
            x_d: (batch_size, num_rays, num_levels, 3). The intersection points in deformed space.
            x_c: (batch_size, num_rays, num_levels, 3). The intersection points in canonical space.
            level: (batch_size, num_rays, num_levels). The manifold level at intersections.
            dist: (batch_size, num_rays, num_levels). The d in formula ray(d) = ray_o + d * ray_d.
            mask_manifold: bool of (batch_size, num_rays, num_levels). Whether the manifold 
                intersection points are valid.
            mask_valid: bool of (batch_size, num_rays, num_levels). Whether the canonical
                correspondence of deformed points are valid.
            rays_o: (batch_size, num_rays, 3). The origin of the rays.
            rays_d: (batch_size, num_rays, 3). The direction of the rays.
        """
        batch_size, num_rays, _ = uv.shape
        num_levels = len(self.manifold_levels)
        # rays_o and rays_d has gradiennt w.r.t. intrinsic and extrinsic.
        rays_o, rays_d = uv_to_rays(uv, intrinsic, extrinsic)  # (batch_size, num_rays, 3)
        rays_o = rays_o[:, None, :].expand(-1, num_rays, -1)  # (batch_size, num_rays, 3)
        rays_idx = torch.arange(batch_size, device=uv.device)[:, None].expand(-1, num_rays)

        # ray marching to find the intersection distance of surface points.
        # we can easily get intersection points in deformed space with distance,
        # thus getting intersection points in the canonical space.
        # autodiff is disabled in this process to ensure low memory usage.
        with torch.no_grad():
            dist, mask_intersect = self.ray_march_manifold(
                manifold_fn,
                rays_o,
                rays_d,
                rays_idx,
                mask,
                self.manifold_levels,
                self.manifold_samples,
                perturb=self.training and self.perturb_when_training,
            )  # (batch_size, num_rays, num_levels)

        # x_d: (batch_size, num_rays, num_levels, 3). The surface points in deformed space.
        # it has gradient w.r.t. rays_o and rays_d, which can be used to optimize camera pose.
        x_d = rays_o[:, :, None, :] + rays_d[:, :, None, :] * dist[..., None]

        # find the canonical correspondences of the exact deformed points
        # x_c has gradients w.r.t. x_d, deformer net params
        # level has gradients w.r.t. x_d, deformer net params, manifold net params
        level, x_c, mask_valid = manifold_fn(
            x_d.reshape(-1, 3).requires_grad_(True),
            rays_idx[:, :, None].expand(-1, -1, num_levels).reshape(-1))
        level = level.reshape(batch_size, num_rays, num_levels)
        x_c = x_c.reshape(batch_size, num_rays, num_levels, 3)
        mask_valid = mask_valid.reshape(batch_size, num_rays, num_levels)
        mask_manifold = mask_intersect & mask_valid  # (batch_size, num_rays, num_levels)

        return x_d, x_c, level, dist, mask_manifold, mask_valid, rays_o, rays_d
