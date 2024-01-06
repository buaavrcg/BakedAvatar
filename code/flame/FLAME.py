"""
Code is adopted from https://github.com/zhengyuf/IMavatar
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os

from flame.lbs import *

FLAME_DIR = os.path.dirname(__file__)
FLAME_MODEL_PATH = os.path.join(FLAME_DIR, "FLAME2020", "generic_model.pkl")


def to_tensor(array, dtype=torch.float32, np_dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return torch.tensor(np.array(array, dtype=np_dtype), dtype=dtype)


class Struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)


class FLAME(nn.Module):
    """
    borrowed from https://github.com/soubhiksanyal/FLAME_PyTorch/blob/master/FLAME.py
    Given flame parameters this class generates a differentiable FLAME function
    which outputs the a mesh and 2D/3D facial landmarks
    """
    def __init__(self,
                 n_shape: int,
                 n_exp: int,
                 shape_params: torch.Tensor,
                 canonical_exp: torch.Tensor,
                 ghostbone=False):
        super(FLAME, self).__init__()
        print(f"creating the FLAME Decoder, n_shape: {n_shape}, n_exp: {n_exp}")
        assert shape_params.shape[-1] == n_shape
        assert os.path.exists(FLAME_MODEL_PATH), f"FLAME model not found at {FLAME_MODEL_PATH}"
        with open(FLAME_MODEL_PATH, 'rb') as f:
            flame_model = Struct(**pickle.load(f, encoding='latin1'))
        factor = 4

        self.dtype = torch.float32
        self.register_buffer('faces_tensor',
                             to_tensor(flame_model.f, dtype=torch.long, np_dtype=np.int64))
        # The vertices of the template model
        self.register_buffer('v_template',
                             to_tensor(flame_model.v_template, dtype=self.dtype) * factor)

        # The shape components
        self.n_shape = n_shape
        shapedirs = to_tensor(flame_model.shapedirs, dtype=self.dtype)
        # There are total 300 shape parameters and 100 expression parameters to control FLAME
        shapedirs_shape = shapedirs[:, :, :n_shape]  # Take first n_shape from shape params
        shapedirs_exp = shapedirs[:, :, 300:300 + n_exp]  # Take first n_exp from exp params
        self.register_buffer('shapedirs', torch.cat([shapedirs_shape, shapedirs_exp], 2) * factor)

        # Pre-blend shape parameters into v_template
        self.v_template.add_(
            blend_shapes(shape_params.unsqueeze(0), self.shapedirs[:, :, :n_shape]).squeeze(0))

        # The pose components
        j_regressor = to_tensor(flame_model.J_regressor, dtype=self.dtype)
        self.register_buffer('J_regressor', j_regressor)
        # pose blend shape basis
        num_pose_basis = flame_model.posedirs.shape[-1]
        posedirs = np.reshape(flame_model.posedirs, [-1, num_pose_basis]).T
        self.register_buffer('posedirs', to_tensor(posedirs, dtype=self.dtype) * factor)
        # indices of parents for each joints
        parents = to_tensor(flame_model.kintree_table[0]).long()
        parents[0] = -1
        self.register_buffer('parents', parents)
        self.register_buffer('lbs_weights', to_tensor(flame_model.weights, dtype=self.dtype))

        # Eyeball and neck rotation
        default_eyball_pose = torch.zeros([1, 6], dtype=self.dtype, requires_grad=False)
        self.register_buffer('eye_pose', default_eyball_pose, False)
        default_neck_pose = torch.zeros([1, 3], dtype=self.dtype, requires_grad=False)
        self.register_buffer('neck_pose', default_neck_pose, False)

        # Canonical pose, expression, vertices, pose_feature, transformations
        self.register_buffer('canonical_pose', torch.zeros(1, 15, dtype=self.dtype), False)
        self.canonical_pose[:, 6] = 0.4  # slightly opened jaw
        self.register_buffer('canonical_exp', canonical_exp.to(dtype=self.dtype), False)
        canonical_verts, canonical_pose_feature, canonical_transforms = \
            self.forward(expression_params=self.canonical_exp, pose_params=self.canonical_pose)
        self.register_buffer('canonical_verts', canonical_verts, False)
        self.register_buffer('canonical_pose_feature', canonical_pose_feature, False)
        if ghostbone:
            ghostbone_transform = torch.eye(4, dtype=self.dtype)[None, None, :, :]
            canonical_transforms = torch.cat([ghostbone_transform, canonical_transforms], 1)
            self.register_buffer('ghostbone_transform', ghostbone_transform, False)
        self.register_buffer('canonical_transforms', canonical_transforms, False)

    def forward(self, expression_params, pose_params):
        """
        FLAME mesh morphing.

        Args:
            expression_params: N X number of expression parameters
            pose_params: N X number of pose parameters (5x3=15)
        Returns:
            vertices: N X V X 3
            pose_feature: torch.tensor N X (4x9)
            transforms: torch.tensor N X 5 X 4 X 4
        """
        batch_size = expression_params.shape[0]

        # Use zero shape parameters here as v_template already considered shape deformation
        shape_params = torch.zeros(batch_size,
                                   self.n_shape,
                                   device=expression_params.device,
                                   dtype=expression_params.dtype)
        betas = torch.cat([shape_params, expression_params], dim=1)

        template_vertices = self.v_template.unsqueeze(0).expand(batch_size, -1, -1)
        vertices, _, pose_feature, transforms = lbs(betas, pose_params, template_vertices,
                                                    self.shapedirs, self.posedirs, self.J_regressor,
                                                    self.parents, self.lbs_weights)

        if hasattr(self, 'ghostbone_transform'):
            transforms = torch.cat(
                [self.ghostbone_transform.expand(transforms.shape[0], -1, -1, -1), transforms], 1)

        return vertices, pose_feature, transforms

    def forward_original_points(self, points_o, normals_o, betas, transforms, pose_feature,
                                shapedirs, posedirs, lbs_weights):
        """
        Transform original points to the deformed space.
        Args:
            points_o: (N, 3). Original positions.
            normals_o: (N, 3). Original normals.
            betas: (N, 50). Expression coefficients.
            transforms: (N, 5 or 6, 4, 4). Bone transformations.
            pose_feature: (N, 36). Pose feature.
            shapedirs: (N, 3, 50). Expression offset vectors.
            posedirs: (N, 36, 3). Pose related offset vectors.
            lbs_weights: (N, 5 or 6). Weights for linear blend skinning.
        Returns:
            points_p: (N, 3). Deformed positions.
        """
        assert len(points_o.shape) == 2 and points_o.shape[1] == 3
        num_points = points_o.shape[0]
        if num_points == 0:
            return points_o

        points_p = forward_pts(
            points_o,
            betas,
            transforms,
            pose_feature,
            shapedirs,
            posedirs,
            lbs_weights,
        )

        # use an approximate normal transformation
        delta = 1e-4
        normals_p = forward_pts(
            points_o + delta * normals_o,
            betas,
            transforms,
            pose_feature,
            shapedirs,
            posedirs,
            lbs_weights,
        ) - points_p
        normals_p = nn.functional.normalize(normals_p, dim=-1)

        return points_p, normals_p

    def forward_points(self, points_c, betas, transforms, pose_feature, shapedirs, posedirs,
                       lbs_weights):
        """
        Transform points from the canonical space to the deformed space.
        Args:
            points_c: (N, 3). Canonical positions.
            betas: (N, 50). Expression coefficients.
            transforms: (N, 5 or 6, 4, 4). Bone transformations.
            pose_feature: (N, 36). Pose feature.
            shapedirs: (N, 3, 50). Expression offset vectors.
            posedirs: (N, 36, 3). Pose related offset vectors.
            lbs_weights: (N, 5 or 6). Weights for linear blend skinning.
        Returns:
            points_p: (N, 3). Deformed positions.
        """
        assert len(points_c.shape) == 2 and points_c.shape[1] == 3
        num_points = points_c.shape[0]
        if num_points == 0:
            return points_c

        points_c_original = inverse_pts(
            points_c,
            self.canonical_exp.expand(num_points, -1),
            self.canonical_transforms.expand(num_points, -1, -1, -1),
            self.canonical_pose_feature.expand(num_points, -1),
            shapedirs,
            posedirs,
            lbs_weights,
        )
        points_p = forward_pts(
            points_c_original,
            betas,
            transforms,
            pose_feature,
            shapedirs,
            posedirs,
            lbs_weights,
        )
        return points_p

    def inverse_skinning_points(self, points_p, transforms, lbs_weights):
        """Transform points from the current pose to the canonical pose."""
        assert len(points_p.shape) == 2 and points_p.shape[1] == 3
        num_points = points_p.shape[0]
        if num_points == 0:
            return points_p

        points_c_original = inverse_skinning_pts(
            points_p,
            transforms,
            lbs_weights,
        )
        points_c = forward_skinning_pts(
            points_c_original,
            self.canonical_transforms.expand(num_points, -1, -1, -1),
            lbs_weights,
        )
        return points_c

    def get_original_points(self, points_c, normals_c, shapedirs, posedirs, lbs_weights):
        """
        Transform points from the canonical space to the original FLAME space.
        Args:
            points_c: (N, 3). Canonical positions.
            normals_c: (N, 3). Canonical normals.
            shapedirs: (N, 3, 50). Expression offset vectors.
            posedirs: (N, 36, 3). Pose related offset vectors.
            lbs_weights: (N, 5 or 6). Weights for linear blend skinning.
        Returns:
            points_original: (N, 3). Original FLAME positions.
            normals_original: (N, 3). Original FLAME normals.
        """
        assert len(points_c.shape) == 2 and points_c.shape[1] == 3
        assert points_c.shape == normals_c.shape
        num_points = points_c.shape[0]
        assert num_points > 0

        points_original = inverse_pts(
            points_c,
            self.canonical_exp.expand(num_points, -1),
            self.canonical_transforms.expand(num_points, -1, -1, -1),
            self.canonical_pose_feature.expand(num_points, -1),
            shapedirs,
            posedirs,
            lbs_weights,
        )

        # just use an approximate normal transformation here
        # shoule work well in most case
        delta = 1e-4
        normals_c = nn.functional.normalize(normals_c, dim=-1)
        normals_original = inverse_pts(
            points_c + delta * normals_c,
            self.canonical_exp.expand(num_points, -1),
            self.canonical_transforms.expand(num_points, -1, -1, -1),
            self.canonical_pose_feature.expand(num_points, -1),
            shapedirs,
            posedirs,
            lbs_weights,
        ) - points_original
        normals_original = nn.functional.normalize(normals_original, dim=-1)

        return points_original, normals_original

    def get_blendshape_offsets(self, betas, pose_feature, shapedirs, posedirs):
        """Get blendshape offsets of the current pose and shape."""
        shape_offsets = torch.einsum('ml,mkl->mk', betas, shapedirs)
        pose_offsets = torch.einsum('mi,mik->mk', pose_feature, posedirs)
        return shape_offsets + pose_offsets
