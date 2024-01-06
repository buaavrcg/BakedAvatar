import torch
import pytorch3d.ops as ops

from flame.FLAME import FLAME
from model.deformer_network import batch_jacobian
from model.renderer import ManifoldRenderer

from utils.render_util import alpha_integration
from utils.training_util import chunked_forward
from utils.misc_util import construct_class_by_name


class DMMavatar(torch.nn.Module):
    def __init__(
        self,
        dim_expression: int,
        dim_frame_latent: int,
        dim_radiance_feature: int,
        dim_pos_feature: int,
        num_texture_basis: int,
        shape_params: torch.Tensor,
        canonical_exp: torch.Tensor,
        ghostbone: bool = False,
        manifold_network_class="model.manifold_network.ManifoldNetwork",
        deformer_network_class="model.deformer_network.ForwardDeformer",
        radiance_network_class="model.radiance_network.RadianceNetwork",
        radiance_decoder_class="model.radiance_network.RadianceDecoder",
        manifold_network_args: dict = {},
        deformer_network_args: dict = {},
        radiance_network_args: dict = {},
        radiance_decoder_args: dict = {},
        manifold_renderer_args: dict = {},
    ):
        super().__init__()
        self.dim_expression = dim_expression
        self.dim_frame_latent = dim_frame_latent
        self.dim_radiance_feature = dim_radiance_feature
        self.dim_pos_feature = dim_pos_feature
        self.num_texture_basis = num_texture_basis
        self.query_batch_size = 100000

        self.ghostbone = ghostbone
        self.flame = FLAME(100, dim_expression, shape_params, canonical_exp, ghostbone)
        self.manifold_net = construct_class_by_name(
            class_name=manifold_network_class,
            **manifold_network_args,
        )
        self.deformer_net = construct_class_by_name(
            self.flame,
            dim_expression,
            class_name=deformer_network_class,
            ghostbone=ghostbone,
            **deformer_network_args,
        )
        self.radiance_net = construct_class_by_name(
            dim_expression,
            dim_frame_latent,
            dim_radiance_feature,
            dim_pos_feature,
            num_texture_basis,
            class_name=radiance_network_class,
            **radiance_network_args,
        )
        if radiance_decoder_class:
            self.radiance_decoder = construct_class_by_name(
                dim_radiance_feature,
                class_name=radiance_decoder_class,
                **radiance_decoder_args,
            )
        else:
            self.radiance_decoder = None
        self.renderer = ManifoldRenderer(**manifold_renderer_args)

    def query_manifold(self, x_d, betas, transforms, pose_feature):
        """
        Query the deformable manifold network at deformed points x_d.
        Args:
            x_d: (N, 3). Points in the deformed space.
            betas: (N, dim_expression). Expression parameters.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
            pose_feature: (N, 36). Pose coefficients for pose-related blendshapes.
        Returns:
            s: (N,). Scalar levels of the manifold at the canonical correspondences.
            x_c: (N, 3). Canonical correspondences of the deformed points x_d.
            mask: (N,). Mask of valid canonical points.
        """
        # find the possible canonical correspondence x_c of deformed points
        # x_c: (N, n_inits, 3) without gradient; mask: (N, n_inits)
        x_c, mask = self.deformer_net.forward_inverse(x_d, betas, transforms, pose_feature)

        # evaluate manifold network at the canonical correspondences
        s = torch.full(mask.shape, 1e3, device=x_c.device)  # (N, n_inits)
        s[mask] = self.manifold_net(x_c[mask]).squeeze(1)  # (n_valid_points,)

        # aggregate manifold level as the minimum level of all the initializations, where
        # minimum level corresponds to the largest occupancy of all the canonical points.
        s, index = torch.min(s, dim=1)  # (N,), (N,)
        x_c = torch.gather(x_c, 1, index[:, None, None].expand(-1, -1, 3)).squeeze(1)  # (N, 3)
        mask = torch.gather(mask, 1, index[:, None]).squeeze(1)  # (N,)

        if x_d.requires_grad or betas.requires_grad \
            or transforms.requires_grad or pose_feature.requires_grad:
            # x_c has gradients w.r.t. x_d, deformer net params and FLAME condition
            x_c[mask] = self.deformer_net.get_differential_canonical_points(
                x_c[mask], betas[mask], transforms[mask], pose_feature[mask], x_d[mask])
            # s has gradients w.r.t. x_d, deformer/manifold net params and FLAME condition
            s[mask] = self.manifold_net(x_c[mask]).squeeze(1)

        return s, x_c, mask

    def forward(self, inputs):
        """
        Args are in a dict of inputs:
            expression: (batch_size, dim_expression). Expression parameters in FLAME.
            pose: (batch_size, 15). Pose parameters in FLAME.
            frame_latent: (batch_size, dim_frame_latent). Per-frame latent code.
            background_rgb: (batch_size, 3). Background color.
        Returns a dictionary of outputs:
            rgb: (batch_size, num_rays, 3). Final rgb output.
            alpha: (batch_size, num_rays). Final alpha output.
            depth: (batch_size, num_rays). Alpha weighted average of depth.
            mask: (batch_size, num_rays). Predicted mask of the object.
            x_d: (batch_size, num_rays, num_levels, 3). Intersections in deformed space.
            x_c: (batch_size, num_rays, num_levels, 3). Intersections in canonical space.
            mask_valid: (batch_size, num_rays, num_levels). Valid mask of the canonical points.
            mask_manifold: (batch_size, num_rays, num_levels). Valid mask of manifold levels.
            alpha_manifold: (batch_size, num_rays, num_levels). Alpha of manifold levels.
            normal: (batch_size, num_rays, num_levels, 3). Normal of manifold levels.
            dist: (batch_size, num_rays, num_levels). Intersection distance in deformed space.
            level: (batch_size, num_rays, num_levels). Manifold level of the intersections.
            level_target: (batch_size, num_rays, num_levels). Target manifold level.
        """
        expression = inputs['expression']  # (batch_size, dim_expression)
        pose = inputs['pose']  # (batch_size, 15)
        if 'frame_latent' in inputs:
            frame_latent = inputs['frame_latent']  # (batch_size, dim_frame_latent)
        else:
            frame_latent = torch.zeros(expression.shape[0],
                                       self.dim_frame_latent,
                                       device=expression.device)

        # get FLAME pose feature and transforms from expression and pose parameters
        # pose_feature: (batch_size, 36); transforms: (batch_size, num_joints, 4, 4)
        _, pose_feature, transforms = self.flame(expression, pose)

        # find manifold intersections (order of dist from near to far)
        x_d, x_c, level, dist, mask_manifold, mask_valid, rays_o, rays_d = self.renderer(
            chunked_forward(
                lambda points, idx: self.query_manifold(points, expression[idx], \
                transforms[idx], pose_feature[idx]), self.query_batch_size), **inputs)

        # get the differential of manifold intersections w.r.t. deformer/manifold net params.
        e_masked = expression[:, None, None, :].expand(*x_c.shape[:3], -1)[mask_manifold]
        t_masked = transforms[:, None, None, ...].expand(*x_c.shape[:3], -1, -1, -1)[mask_manifold]
        p_masked = pose_feature[:, None, None, :].expand(*x_c.shape[:3], -1)[mask_manifold]
        if self.training:
            x_c_manifold = self.get_differential_manifold_intersections(
                x_c[mask_manifold],
                e_masked,  # (n_valid_points, dim_expression)
                t_masked,
                p_masked,
                rays_o[:, :, None, :].expand(*x_c.shape[:3], -1)[mask_manifold],
                rays_d[:, :, None, :].expand(*x_c.shape[:3], -1)[mask_manifold],
                self.renderer.manifold_levels[None, None, :].expand(*x_c.shape[:3])[mask_manifold],
            )  # (n_valid_points, 3)
        else:
            x_c_manifold = x_c[mask_manifold]  # (n_valid_points, 3)

        # query FLAME blendshapes and lbs weights at the canonical points
        shapedirs, posedirs, lbs_weights = self.deformer_net.query_weights(x_c.reshape(-1, 3))
        shapedirs = shapedirs.reshape(*x_c.shape[:3], 3, self.dim_expression)
        posedirs = posedirs.reshape(*x_c.shape[:3], 36, 3)
        lbs_weights = lbs_weights.reshape(*x_c.shape[:3], -1)

        # also get ground truth blendshapes and lbs weights
        flame_distances, flame_index, flame_shapedirs, flame_posedirs, \
            flame_lbs_weights = self.get_flame_weights(x_c_manifold, mask_manifold)

        # get the normal direction of manifold intersections
        # normal: (batch_size, num_rays, num_levels, 3)
        normal = torch.zeros(*x_c.shape, device=x_c.device)
        normal[mask_manifold] = self.get_manifold_gradient(x_c_manifold, e_masked, t_masked,
                                                           p_masked)
        normal[mask_manifold] = torch.nn.functional.normalize(normal[mask_manifold], dim=-1)

        # evaluate radiance network at valid manifold intersections
        # feature: (batch_size, num_rays, num_levels, dim_radiance_feature)
        # alpha: (batch_size, num_rays, num_levels) alpha values in [0, 1]
        # pos_feature: (batch_size, num_rays, num_levels, dim_pos_feature)
        # texture_coef: (batch_size, num_rays, num_levels, num_texture_basis)
        feature = torch.zeros(*x_c.shape[:3], self.dim_radiance_feature, device=x_c.device)
        alpha = torch.zeros(*x_c.shape[:3], device=x_c.device)
        pos_feature = torch.zeros(*x_c.shape[:3], self.dim_pos_feature, device=x_c.device)
        texture_coef = torch.zeros(*x_c.shape[:3], self.num_texture_basis, device=x_c.device)
        feature[mask_manifold], alpha[mask_manifold], pos_feature[mask_manifold], \
            texture_coef[mask_manifold], _ = \
            self.radiance_net(
                x=x_c_manifold,
                normal=normal[mask_manifold],
                viewdir=rays_d[:, :, None, :].expand(*x_c.shape[:3], -1)[mask_manifold],
                exp=e_masked,
                pose=pose[:, None, None, 6:15].expand(*x_c.shape[:3], -1)[mask_manifold],
                latent=frame_latent[:, None, None, :].expand(*x_c.shape[:3], -1)[mask_manifold],
            )  # (n_valid_points, dim_radiance_feature), (n_valid_points,)

        # integrate radiance features along the rays using alpha compositing
        # feature: (batch_size, num_rays, dim_radiance_feature)
        # weight_bg: (batch_size, num_rays)
        (feature_sum, ), depth, weight_bg = alpha_integration(
            (feature, ), dist, alpha, mask_manifold, self.renderer.far_distance)

        # decode accumulated radiance features to rgb output
        if self.radiance_decoder is None:
            assert self.dim_radiance_feature == 3, \
                "radiance decoder is not used, but dim_radiance_feature is not 3"
            rgb = feature_sum
        else:
            rgb = self.radiance_decoder(
                feature_sum.reshape(-1, self.dim_radiance_feature),
                rays_d.reshape(-1, 3),
            ).reshape(*feature_sum.shape[:-1], 3)  # (batch_size, num_rays, 3)

        # blend background color
        rgb = rgb * (1 - weight_bg[:, :, None]) \
            + inputs['background_rgb'][:, None, :] * weight_bg[:, :, None]

        return {
            "rgb": rgb,
            "alpha": 1 - weight_bg,
            "depth": depth,
            "mask": torch.any(mask_manifold, dim=-1),
            "x_d": x_d,
            "x_c": x_c,
            "mask_valid": mask_valid,
            "mask_manifold": mask_manifold,
            "alpha_manifold": alpha,
            "feature_manifold": feature,
            "feature_sum": feature_sum,
            "pos_feature": pos_feature,
            "texture_coef": texture_coef,
            "normal": normal,
            "dist": dist,
            "level": level,
            "level_target": self.renderer.manifold_levels[None, None, :].expand_as(level),
            "img_res": inputs['img_res'],
            "shapedirs": shapedirs,
            "posedirs": posedirs,
            "lbs_weights": lbs_weights,
            "flame_distances": flame_distances,
            "flame_index": flame_index,
            "flame_shapedirs": flame_shapedirs,
            "flame_posedirs": flame_posedirs,
            "flame_lbs_weights": flame_lbs_weights,
        }

    def get_differential_manifold_intersections(self, x_c, betas, transforms, pose_feature, rays_o,
                                                rays_d, manifold_levels):
        """
        Get a differential version of canonical manifold intersection points
        w.r.t. the deformer/manifold net params.
        Args:
            x_c: (N, 3). Manifold intersection points in canonical space.
            betas: (N, dim_expression). Expression parameters.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
            pose_feature: (N, 36). Pose coefficients for pose-related blendshapes.
            rays_o: (N, 3). The origin of the rays.
            rays_d: (N, 3). The direction of the rays.
            manifold_levels: (N,). The target manifold level of each intersection point.
        Returns:
            x_c_with_grad: (N, 3). x_c with gradients w.r.t. deformer/manifold net params.
        """
        if x_c.shape[0] == 0:
            return x_c.detach()

        x_c = x_c.detach().requires_grad_(True)
        x_d = self.deformer_net(x_c, betas, transforms, pose_feature)  # (N, 3)
        scalar = self.manifold_net(x_c)  # (N, 1)
        scalar_diff = scalar - manifold_levels[:, None]

        points_dir = x_d - rays_o  # (N, 3)
        # points_dir = points_dir / torch.norm(points_dir, dim=1, keepdim=True)
        cross_product = torch.cross(rays_d, points_dir, dim=1)  # (N, 3)

        constant = torch.cat([cross_product[:, 0:2], scalar_diff], dim=1)
        grad_inv = batch_jacobian(constant, x_c).inverse()
        correction = constant - constant.detach()
        correction = -torch.einsum('bij,bj->bi', grad_inv, correction)

        # adding implicit diff to autodiff: x_c_with_grad = x_c + 0 and x_c' = correction'
        x_c_with_grad = x_c.detach() + correction

        return x_c_with_grad

    def get_manifold_gradient(self, x_c, betas, transforms, pose_feature):
        """
        Get the gradient of the manifold w.r.t. deformed manifold intersection points x_d.
        Args:
            x_c: (N, 3). Manifold intersection points in canonical space.
            betas: (N, dim_expression). Expression parameters.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
            pose_feature: (N, 36). Pose coefficients for pose-related blendshapes.
        Returns:
            gradient: (N, 3). The gradient of the manifold net at manifold intersection points.
        """
        if x_c.shape[0] == 0:
            return torch.zeros_like(x_c)

        @torch.enable_grad()
        def forward_gradient(x_c, betas, transforms, pose_feature):
            x_c = x_c.detach().requires_grad_(True)
            x_d = self.deformer_net.forward(x_c, betas, transforms, pose_feature)
            grad_inv = batch_jacobian(x_d, x_c, False, True).inverse()
            scalar = self.manifold_net(x_c)
            gradient, = torch.autograd.grad(
                outputs=scalar,
                inputs=x_c,
                grad_outputs=torch.ones_like(scalar, requires_grad=False),
                create_graph=self.training,
                retain_graph=self.training,
            )
            gradient = torch.einsum('nij,ni->nj', grad_inv, gradient)
            return gradient

        if self.training:
            return forward_gradient(x_c, betas, transforms, pose_feature)
        else:
            forward_gradient = chunked_forward(forward_gradient, self.query_batch_size)
            gradient, = forward_gradient(x_c, betas, transforms, pose_feature)
            return gradient

    def get_flame_weights(self, x_c_manifold, mask_manifold):
        """
        Get ground truth weights of the FLAME model at manifold intersection points.
        Args:
            x_c_manifold: (num_points, 3). Manifold intersection points in canonical space.
            mask_manifold: (batch_size, num_rays, num_levels). Mask of manifold intersections.
        Returns:
            flame_distances: (batch_size, num_rays, num_levels). Distance to the nearest vertex.
            flame_index: (batch_size, num_rays, num_levels). Index of the nearest vertex.
            flame_shapedirs: (batch_size, num_rays, num_levels, 3, 50).
            flame_posedirs: (batch_size, num_rays, num_levels, 36, 3).
            flame_lbs_weights: (batch_size, num_rays, num_levels, 5).
        """
        flame_distances = torch.zeros_like(mask_manifold, dtype=torch.float32)
        flame_index = torch.ones_like(mask_manifold, dtype=torch.long)

        flame_distances_manifold, flame_index_manifold, _ = \
            ops.knn_points(x_c_manifold.unsqueeze(0), self.flame.canonical_verts, K=1)
        flame_distances[mask_manifold] = flame_distances_manifold.squeeze(-1).squeeze(0)
        flame_index[mask_manifold] = flame_index_manifold.squeeze(-1).squeeze(0)

        flame_shapedirs = self.flame.shapedirs[flame_index, :, 100:]
        flame_posedirs = self.flame.posedirs.reshape(36, -1, 3).swapaxes(0, 1)[flame_index]
        flame_lbs_weights = self.flame.lbs_weights[flame_index]
        if self.ghostbone:
            flame_lbs_weights = torch.cat(
                [torch.zeros_like(flame_lbs_weights[..., :1]), flame_lbs_weights], dim=-1)

        return flame_distances, flame_index, flame_shapedirs, flame_posedirs, flame_lbs_weights

    def query_canonical_manifold(self,
                                 x_c,
                                 no_gradient=False,
                                 create_graph=False,
                                 retain_graph=True):
        """
        Get the gradient of the manifold w.r.t. canonical manifold intersection points x_c.
        Args:
            x_c: (N, 3). Manifold intersection points in canonical space.
            no_gradient: If True, only return the level set value of the manifold.
        Returns:
            scalar: (N, 1). The level set value of the manifold at manifold intersection points.
            gradient: (N, 3). The gradient of the manifold at manifold intersection points.
        """
        assert x_c.shape[0] > 0 and x_c.shape[1] == 3
        if no_gradient:
            return self.manifold_net(x_c)

        with torch.enable_grad():
            if not x_c.requires_grad:
                x_c = x_c.detach().requires_grad_(True)
            scalar = self.manifold_net(x_c)
            gradient, = torch.autograd.grad(
                outputs=scalar,
                inputs=x_c,
                grad_outputs=torch.ones_like(scalar, requires_grad=False),
                create_graph=create_graph,
                retain_graph=retain_graph,
            )

        return scalar, gradient
