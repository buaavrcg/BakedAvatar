import torch
import torch.nn.functional as F
import torchmetrics.functional as metrics
from pytorch3d.loss import mesh_normal_consistency, mesh_laplacian_smoothing, chamfer_distance


class Loss(torch.nn.Module):
    def __init__(self,
                 rgb_loss_weight,
                 mask_loss_weight,
                 flame_loss_weight=0.0,
                 flame_distance_loss_weight=0.0,
                 mask_alpha=10.0,
                 mask_target_gamma=0.0,
                 flame_dist_threshold=0.001,
                 flame_target_gamma=0.0,
                 flame_no_cloth_loss=False,
                 flame_loss_decay_iterations=0,
                 flame_loss_decay_factor=0.5,
                 use_semantic=False):
        super().__init__()
        self.rgb_loss_weight = rgb_loss_weight
        self.mask_loss_weight = mask_loss_weight
        self.flame_loss_weight = flame_loss_weight
        self.flame_distance_loss_weight = flame_distance_loss_weight
        self.mask_alpha = mask_alpha
        self.mask_target_gamma = mask_target_gamma
        self.flame_dist_threshold = flame_dist_threshold
        self.flame_target_gamma = flame_target_gamma
        self.flame_no_cloth_loss = flame_no_cloth_loss
        self.flame_loss_decay_iterations = flame_loss_decay_iterations
        self.flame_loss_decay_factor = flame_loss_decay_factor
        self.use_semantic = use_semantic

    def rgb_loss(self, rgb_output, rgb_gt, mask_gt):
        """
        Args:
            rgb_output: (batch_size, num_rays, 3). Output RGB values.
            rgb_gt: (batch_size, num_rays, 3). Ground truth RGB values.
            mask_gt: (batch_size, num_rays). Ground truth object mask.
        Returns:
            rgb_loss: (1,). RGB loss.
        """
        return F.l1_loss(rgb_output[mask_gt], rgb_gt[mask_gt], reduction='mean')

    def mask_loss(self, level_output, level_target, mask_valid, mask_output, mask_gt):
        """
        Args:
            level_output: (batch_size, num_rays, num_levels). Output manifold levels.
            level_target: (batch_size, num_rays, num_levels). Target manifold levels.
            mask_valid: (batch_size, num_rays, num_levels). Mask of valid level outputs.
            mask_output: (batch_size, num_rays, num_levels). Output object mask.
            mask_gt: (batch_size, num_rays). Ground truth object mask.
        Returns:
            mask_loss: (1,). Mask loss.
        """
        mask_gt = mask_gt[:, :, None].expand_as(mask_valid)

        # Only supervise non-surface points
        mask_outside = mask_valid & ~(mask_output & mask_gt)
        assert mask_outside.any(), f"valid rate: {mask_valid.float().mean()}"

        level_diff = self.mask_alpha * (level_output[mask_outside] - level_target[mask_outside])
        occupancy_target = mask_gt[mask_outside].float()
        loss = F.binary_cross_entropy_with_logits(-level_diff, occupancy_target, reduction='none')

        if self.mask_target_gamma != 0:
            level_target_diff = torch.abs(level_target[mask_outside])
            loss_weight = torch.exp(-self.mask_target_gamma * level_target_diff)
            loss_weight = loss_weight / loss_weight.mean()  # let weight mean equal to 1
            loss = (loss * loss_weight).mean()
        else:
            loss = loss.mean()

        return loss / self.mask_alpha

    def flame_loss(self, outputs, mask_output, mask_gt, semantics_gt=None):
        """
        Supervise FLAME weights with the closest point on the FLAME surface.
        Args:
            flame_distances: (batch_size, num_rays, num_levels). Distance of manifold
                intersection points and the closest point on the FLAME surface.
            flame_output: (batch_size, num_rays, num_levels, ...).
            flame_target: (batch_size, num_rays, num_levels, ...).
            mask_output: (batch_size, num_rays, num_levels). Output object mask.
            mask_gt: (batch_size, num_rays). Ground truth object mask.
            semantics_gt: (batch_size, num_rays, 9). Ground truth semantic mask.
        Returns:
            flame_loss: (1,). FLAME loss.
        """
        flame_distances = outputs['flame_distances']
        flame_distance_mask = flame_distances < self.flame_dist_threshold
        mask = mask_output & mask_gt[:, :, None] & flame_distance_mask

        if semantics_gt is not None:
            semantics_gt = semantics_gt[..., None].expand(-1, -1, -1, flame_distances.shape[-1])
            mask_mouth = semantics_gt[:, :, 3]
            mask_cloth = semantics_gt[:, :, 7]
            if self.flame_no_cloth_loss:
                mask = mask & ~(mask_cloth)

        if not mask.any():
            return torch.tensor(0.0, device=flame_distances.device), {}

        # get GT shapedirs, posedirs, lbs_weights
        gt_shapedirs = outputs['flame_shapedirs']
        gt_posedirs = outputs['flame_posedirs']
        gt_lbs_weights = outputs['flame_lbs_weights']

        # adjust FLAME GT according to semantics
        if semantics_gt is not None:
            if not self.flame_no_cloth_loss and gt_lbs_weights.shape[-1] == 6:  # has ghostbone
                # set lbs weights of cloth to ghostbone
                gt_lbs_weights[mask_cloth] = 0.0
                gt_lbs_weights[mask_cloth][:, 0] = 1.0

            # set posedirs and shapedirs of mouth interior to zero
            gt_posedirs[mask_mouth] = 0.0
            gt_shapedirs[mask_mouth] = 0.0

            # set shapedirs of cloth to zero
            gt_shapedirs[mask_cloth] = 0.0

        loss_shapedirs = 1000.0 * F.mse_loss(outputs['shapedirs'][mask], gt_shapedirs[mask])
        loss_posedirs = 1000.0 * F.mse_loss(outputs['posedirs'][mask], gt_posedirs[mask])
        loss_lbs_weights = F.mse_loss(outputs['lbs_weights'][mask], gt_lbs_weights[mask])
        loss_flame = loss_shapedirs + loss_posedirs + 0.1 * loss_lbs_weights
        return loss_flame, {
            'loss_flame': loss_flame.detach(),
            'loss_shapedirs': loss_shapedirs.detach(),
            'loss_posedirs': loss_posedirs.detach(),
            'loss_lbs_weights': loss_lbs_weights.detach()
        }

    def flame_distance_loss(self, flame_distances, mask_output, mask_gt, semantics_gt,
                            level_target):
        """
        Supervise FLAME weights with the closest point on the FLAME surface.
        Args:
            flame_distances: (batch_size, num_rays, num_levels). Distance of manifold
                intersection points and the closest point on the FLAME surface.
            mask_output: (batch_size, num_rays, num_levels). Output object mask.
            mask_gt: (batch_size, num_rays). Ground truth object mask.
            semantics_gt: (batch_size, num_rays, 9). Ground truth semantic mask.
            level_target: (batch_size, num_rays, num_levels). Target manifold levels.
        Returns:
            flame_distance_loss: (1,). FLAME distance loss.
        """
        mask_skin = semantics_gt[:, :, 0]
        mask = mask_output & mask_gt[:, :, None] & mask_skin[:, :, None]

        if not mask.any():
            return torch.tensor(0.0, device=flame_distances.device)

        loss = torch.abs(flame_distances[mask])
        if self.flame_target_gamma != 0:
            level_target_diff = torch.abs(level_target[mask])
            loss_weight = torch.exp(-self.mask_target_gamma * level_target_diff)
            loss_weight = loss_weight / loss_weight.mean()  # let weight mean equal to 1
            loss = (loss * loss_weight).mean()
        else:
            loss = loss.mean()

        return loss

    def calc_statistics(self, outputs, targets):
        batch_size, num_rays, _ = outputs['rgb'].shape
        img_res = outputs['img_res'][0].cpu().tolist()
        is_full_image = img_res[0] * img_res[1] == num_rays
        mask_gt = targets['mask']

        stats_dict = {}
        stats_dict['output_hit'] = torch.mean(outputs['mask'][mask_gt].float())
        stats_dict['manifold_hit'] = torch.mean(outputs['mask_manifold'][mask_gt].float())
        stats_dict['valid_rate'] = torch.mean(outputs['mask_valid'][mask_gt].float())

        # statistics for full-image
        if is_full_image:
            rgb_image_output = outputs['rgb'].detach()
            rgb_image_output = rgb_image_output.view(batch_size, *img_res, 3).permute(0, 3, 1, 2)
            rgb_image_gt = targets['rgb'].view(batch_size, *img_res, 3).permute(0, 3, 1, 2)

            stats_dict['psnr'] = metrics.peak_signal_noise_ratio(rgb_image_output, rgb_image_gt)
            stats_dict['ssim'] = metrics.structural_similarity_index_measure(
                rgb_image_output, rgb_image_gt)

        return stats_dict

    def forward(self, outputs, targets, iteration):
        mask_gt = targets['mask']

        loss_rgb = self.rgb_loss(outputs['rgb'], targets['rgb'], mask_gt)
        loss_mask = self.mask_loss(outputs['level'], outputs['level_target'], outputs['mask_valid'],
                                   outputs['mask_manifold'], mask_gt)
        loss_total = self.rgb_loss_weight * loss_rgb + self.mask_loss_weight * loss_mask
        loss_dict = {'loss_rgb': loss_rgb.detach(), 'loss_mask': loss_mask.detach()}

        flame_loss_scale = 1.0
        if self.flame_loss_decay_iterations > 0:
            exponent = iteration // self.flame_loss_decay_iterations
            flame_loss_scale = self.flame_loss_decay_factor**exponent

        if self.flame_loss_weight > 0:
            loss_flame, loss_dict_flame = self.flame_loss(
                outputs, outputs['mask_manifold'], mask_gt,
                targets['semantic'] if self.use_semantic else None)
            loss_total += self.flame_loss_weight * flame_loss_scale * loss_flame
            loss_dict.update(loss_dict_flame)

        if self.flame_distance_loss_weight > 0 and self.use_semantic:
            loss_flame_distance = self.flame_distance_loss(outputs['flame_distances'],
                                                           outputs['mask_manifold'], mask_gt,
                                                           targets['semantic'],
                                                           outputs['level_target'])
            loss_total += self.flame_distance_loss_weight * flame_loss_scale * loss_flame_distance
            loss_dict['loss_flame_distance'] = loss_flame_distance.detach()

        loss_dict['loss'] = loss_total.detach()
        stats_dict = self.calc_statistics(outputs, targets)
        return loss_total, loss_dict | stats_dict


class MeshFittingLoss(torch.nn.Module):
    def __init__(self,
                 level_loss_weight=1.0,
                 point_loss_weight=1.0,
                 normal_loss_weight=0.0,
                 vertex_reg_loss_weight=0.0,
                 normal_smooth_loss_weight=0.01,
                 mesh_smooth_loss_weight=0.1,
                 edge_loss_weight=1.0):
        super().__init__()
        self.level_loss_weight = level_loss_weight
        self.point_loss_weight = point_loss_weight
        self.normal_loss_weight = normal_loss_weight
        self.vertex_reg_loss_weight = vertex_reg_loss_weight
        self.normal_smooth_loss_weight = normal_smooth_loss_weight
        self.mesh_smooth_loss_weight = mesh_smooth_loss_weight
        self.edge_loss_weight = edge_loss_weight

    def level_loss(self, scalar_output, level_target):
        """
        Scalar output at vertices should be equal to level target.
        Args:
            scalar_output: (num_levels, num_points). Output scalar values.
            level_target: (num_levels,). Ground truth level values.
        Returns:
            level_loss: (1,). Point level loss.
        """
        return torch.abs(scalar_output - level_target.unsqueeze(1)).mean()

    def point_loss(self, vertices_output, points_target):
        """
        Apply chamfer distance to vertices and target point cloud.
        Args:
            vertices_output: (num_levels, num_points, 3). Output vertices.
            points_target: (num_levels, num_points, 3). Sampled ground truth points.
        Returns:
            point_loss: (1,). Point level loss.
        """
        chamfer_losses = []
        for i, points in enumerate(points_target):
            loss, _ = chamfer_distance(vertices_output[i:i + 1], points.unsqueeze(0))
            chamfer_losses.append(loss)
        return torch.stack(chamfer_losses).mean()

    def normal_loss(self, scalar_output, level_target, normals_output, gradient_target):
        """
        Normals should be equal to the norm of gradient target.
        Args:
            scalar_output: (num_levels, num_points). Output scalar values.
            normals_output: (num_levels, num_points, 3). Vertices normals.
            gradient_target: (num_levels, num_points, 3). Ground truth gradients.
        Returns:
            gradient_loss: (1,). Point gradient loss.
        """
        normals_output = F.normalize(normals_output, dim=-1)
        normals_target = F.normalize(gradient_target, dim=-1)

        weights = 1 / (1 + 100 * torch.abs(scalar_output - level_target.unsqueeze(1)))

        cos_theta = torch.bmm(normals_output.view(-1, 1, 3), normals_target.view(-1, 3, 1))
        cos_theta = cos_theta.reshape(normals_output.shape[:2])
        loss_angle = torch.square(1 - cos_theta) * weights

        return loss_angle.mean()

    def vertex_reg_loss(self, vertices_output, initial_vertices):
        """
        Regularization loss for vertices.
        Args:
            vertices_output: (num_levels, num_points, 3). Output vertices.
            initial_vertices: (num_points, 3). Initial vertices.
        Returns:
            vertex_reg_loss: (1,). Vertex regularization loss.
        """
        return torch.square(vertices_output - initial_vertices[None, :, :]).mean()

    def edge_loss(self,
                  meshes,
                  target_length: float = 0.0,
                  edge_scale: float = 10.0,
                  edge_smoothness: float = 4.0):
        """
        Penalizes the length of edges in the mesh.
        Args:
            meshes: Meshes object with a batch of meshes.
            target_length: Resting value for the edge length.
            edge_scale: Scaling factor for the edge length loss.
            edge_smoothness: Smoothness factor for the edge length loss.
        Returns:
            loss: (1,). Average loss across the batch.
        """
        edges_packed = meshes.edges_packed()  # (sum(E_n), 3)
        verts_packed = meshes.verts_packed()  # (sum(V_n), 3)
        edge_to_mesh_idx = meshes.edges_packed_to_mesh_idx()  # (sum(E_n), )

        weights = meshes.num_edges_per_mesh().gather(0, edge_to_mesh_idx)
        weights = 1.0 / weights.float()

        verts_edges = verts_packed[edges_packed]
        v0, v1 = verts_edges.unbind(1)
        loss = (((v0 - v1).norm(dim=1, p=2) - target_length) * edge_scale)**edge_smoothness
        loss = loss * weights

        return loss.sum() / len(meshes)

    def forward(self, outputs, targets):
        loss_level = self.level_loss(outputs['scalar'], targets['level'])
        loss_total = self.level_loss_weight * loss_level
        loss_dict = {
            'loss_level': loss_level.detach(),
        }

        if self.point_loss_weight > 0:
            loss_point = self.point_loss(outputs['verts_points'], targets['points'])
            loss_total += self.point_loss_weight * loss_point
            loss_dict['loss_point'] = loss_point.detach()

        if self.normal_loss_weight > 0:
            loss_normal = self.normal_loss(outputs['scalar'], targets['level'], outputs['normals'],
                                           outputs['gradient'])
            loss_total += self.normal_loss_weight * loss_normal
            loss_dict['loss_normal'] = loss_normal.detach()

        if self.vertex_reg_loss_weight > 0:
            loss_vertex_reg = self.vertex_reg_loss(outputs['verts_points'],
                                                   targets['initial_vertices'])
            loss_total += self.vertex_reg_loss_weight * loss_vertex_reg
            loss_dict['loss_vertex_reg'] = loss_vertex_reg.detach()

        if self.normal_smooth_loss_weight > 0:
            loss_normal_smooth = mesh_normal_consistency(outputs['meshes'])
            loss_total += self.normal_smooth_loss_weight * loss_normal_smooth
            loss_dict['loss_normal_smooth'] = loss_normal_smooth.detach()

        if self.mesh_smooth_loss_weight > 0:
            loss_mesh_smooth = mesh_laplacian_smoothing(outputs['meshes'])
            loss_total += self.mesh_smooth_loss_weight * loss_mesh_smooth
            loss_dict['loss_mesh_smooth'] = loss_mesh_smooth.detach()

        if self.edge_loss_weight > 0:
            loss_edge = self.edge_loss(outputs['meshes'])
            loss_total += self.edge_loss_weight * loss_edge
            loss_dict['loss_edge'] = loss_edge.detach()

        loss_dict['loss'] = loss_total.detach()
        return loss_total, loss_dict


class FineTuningLoss(torch.nn.Module):
    def __init__(self, rgb_loss_weight=1.0, perceptual_loss_weight=0.1, use_semantic=False):
        super().__init__()
        self.rgb_loss_weight = rgb_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        if perceptual_loss_weight > 0.0:
            from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        self.use_semantic = use_semantic

    def forward(self, outputs, targets):
        batch_size, num_rays, _ = outputs['rgb'].shape
        img_res = outputs['img_res'][0].cpu().tolist()
        is_full_image = img_res[0] * img_res[1] == num_rays
        assert is_full_image, 'Fine-tuning loss only supports full-image training.'

        rgb_image_output = outputs['rgb'].view(batch_size, *img_res, 3).permute(0, 3, 1, 2)
        rgb_image_gt = targets['rgb'].view(batch_size, *img_res, 3).permute(0, 3, 1, 2)
        mask_gt = targets['mask']

        loss_rgb = F.l1_loss(outputs['rgb'][mask_gt], targets['rgb'][mask_gt])
        loss_total = self.rgb_loss_weight * loss_rgb
        loss_dict = {'loss_rgb': loss_rgb.detach()}

        if self.perceptual_loss_weight > 0:
            loss_perceptual = self.lpips(torch.clamp(rgb_image_output, 0.0, 1.0),
                                         torch.clamp(rgb_image_gt, 0.0, 1.0))
            loss_total += self.perceptual_loss_weight * loss_perceptual
            loss_dict['loss_perceptual'] = loss_perceptual.detach()

        # TODO: add other losses
        loss_dict['loss'] = loss_total.detach()

        # statistics for full-image
        stats_dict = {}
        stats_dict['output_hit'] = torch.mean(outputs['mask'][mask_gt].float())
        stats_dict['manifold_hit'] = torch.mean(outputs['mask_manifold'][mask_gt].float())
        stats_dict['psnr'] = metrics.peak_signal_noise_ratio(rgb_image_output, rgb_image_gt)
        stats_dict['ssim'] = metrics.structural_similarity_index_measure(
            rgb_image_output, rgb_image_gt)
        return loss_total, loss_dict | stats_dict
