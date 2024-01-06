"""
Code is adopted from https://github.com/zhengyuf/IMavatar
"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import torch
import torch.nn.functional as F


def lbs(betas, pose, v_template, shapedirs, posedirs, J_regressor, parents, lbs_weights):
    ''' Performs Linear Blend Skinning with the given shape and pose parameters

        Parameters
        ----------
        betas : torch.tensor BxNB
            The tensor of shape parameters
        pose : torch.tensor Bx(J + 1) * 3
            The pose parameters in axis-angle format
        v_template torch.tensor BxVx3
            The template mesh that will be deformed
        shapedirs : torch.tensor 1xNB
            The tensor of PCA shape displacements
        posedirs : torch.tensor Px(V * 3)
            The pose PCA coefficients
        J_regressor : torch.tensor JxV
            The regressor array that is used to calculate the joints from
            the position of the vertices
        parents: torch.tensor J
            The array that describes the kinematic tree for the model
        lbs_weights: torch.tensor N x V x (J + 1)
            The linear blend skinning weights that represent how much the
            rotation matrix of each part affects each vertex

        Returns
        -------
        verts: torch.tensor BxVx3
            The vertices of the mesh after applying the shape and pose
            displacements.
        joints: torch.tensor BxJx3
            The joints of the model
        pose_feature: torch.tensor Bx(4x9)
        transformations: torch.tensor Bx5x4x4
    '''

    batch_size = max(betas.shape[0], pose.shape[0])
    device, dtype = betas.device, betas.dtype

    # Add shape contribution
    v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get the joints
    # NxJx3 array
    J = vertices2joints(J_regressor, v_shaped)

    # 3. Add pose blend shapes
    # N x J x 3 x 3
    ident = torch.eye(3, dtype=dtype, device=device)
    # pose2rot=True
    rot_mats = batch_rodrigues(pose.view(-1, 3)).view(batch_size, -1, 3, 3)

    pose_feature = (rot_mats[:, 1:, :, :] - ident).view(batch_size, -1)
    # (N x P) x (P, V * 3) -> N x V x 3
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)

    v_posed = pose_offsets + v_shaped

    # 4. Get the global joint location
    J_transformed, A = batch_rigid_transform(rot_mats, J, parents)

    # 5. Do skinning:
    # W is N x V x (J + 1)
    W = lbs_weights.unsqueeze(dim=0).expand(batch_size, -1, -1)
    # (N x V x (J + 1)) x (N x (J + 1) x 16)
    num_joints = J_regressor.shape[0]
    T = torch.matmul(W, A.view(batch_size, num_joints, 16)).view(batch_size, -1, 4, 4)

    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))

    verts = v_homo[:, :, :3, 0]

    return verts, J_transformed, pose_feature, A


def forward_pts(pnts_c, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights):
    ''' Transform a set of points from canonical space to deformed space

        Parameters
        ----------
        pnts_c : (num_points, 3)
        betas : (num_points, num_betas)
        pose_feature : (num_points, 36)
        transformations : (num_points, J + 1, 4, 4)
        shapedirs : (num_points, 3, num_betas)
        posedirs : (num_points, 36, 3)
        lbs_weights: (num_points, J + 1)

        Returns
        -------
        pnts_p : (num_points, 3)
    '''
    # Add shape contribution
    pnts_shaped = pnts_c + torch.einsum('ml,mkl->mk', betas, shapedirs)

    # Add pose contribution
    pose_offsets = torch.einsum('mi,mik->mk', pose_feature, posedirs)
    assert pose_offsets.shape == pnts_shaped.shape
    pnts_posed = pnts_shaped + pose_offsets

    # Do linear blend skinning
    pnts_p = forward_skinning_pts(pnts_posed, transformations, lbs_weights)

    return pnts_p


def inverse_pts(pnts_p, betas, transformations, pose_feature, shapedirs, posedirs, lbs_weights):
    ''' Transform a set of points from deformed space to canonical space

        Parameters
        ----------
        pnts_p : (num_points, 3)
        betas : (num_points, num_betas)
        pose_feature : (num_points, 36)
        transformations : (num_points, J + 1, 4, 4)
        shapedirs : (num_points, 3, num_betas)
        posedirs : (num_points, 36, 3)
        lbs_weights: (num_points, J + 1)

        Returns
        -------
        pnts_c : (num_points, 3)
    '''
    # Undo linear blend skinning
    pnts_posed = inverse_skinning_pts(pnts_p, transformations, lbs_weights)

    # Subtract pose contribution
    pose_offsets = torch.einsum('mi,mik->mk', [pose_feature, posedirs])
    assert pose_offsets.shape == pnts_posed.shape
    pnts_shaped = pnts_posed - pose_offsets

    # Subtract shape contribution
    pnts_c = pnts_shaped - torch.einsum('ml,mkl->mk', betas, shapedirs)

    return pnts_c


def forward_skinning_pts(pnts_c, transformations, lbs_weights):
    # pnts_c: [num_points, 3]
    num_points, device, dtype = pnts_c.shape[0], pnts_c.device, pnts_c.dtype

    # Do skinning:
    # W is num_points x (J + 1)
    W = lbs_weights
    num_joints = W.shape[-1]
    # T: [num_points, (J + 1)] x [num_points, (J + 1), 16] --> [num_points, 16]
    T = torch.einsum('mj, mjk->mk',
                     [W, transformations.view(-1, num_joints, 16)]).view(num_points, 4, 4)

    homogen_coord = torch.ones([num_points, 1], dtype=dtype, device=device)
    # v_posed_homo: num_points, 4
    v_homo = torch.cat([pnts_c, homogen_coord], dim=1)
    # v_homo: [num_points, 4, 4] x [num_points, 4, 1] --> [num_points, 4, 1]
    v_homo = torch.matmul(T, torch.unsqueeze(v_homo, dim=-1))
    # pnts: [num_points, 3]
    pnts = v_homo[:, :3, 0]

    return pnts


def inverse_skinning_pts(pnts_p, transformations, lbs_weights):
    # pnts_p: [num_points, 3]
    num_points, device, dtype = pnts_p.shape[0], pnts_p.device, pnts_p.dtype

    # Do skinning:
    # W is num_points x (J + 1)
    W = lbs_weights
    num_joints = W.shape[-1]
    # T: [num_points, (J + 1)] x [num_points, (J + 1), 16] --> [num_points, 16]
    T = torch.einsum('mj, mjk->mk',
                     [W, transformations.view(-1, num_joints, 16)]).view(num_points, 4, 4)

    homogen_coord = torch.ones([num_points, 1], dtype=dtype, device=device)
    # pnts_p: num_points, 4
    pnts_p = torch.cat([pnts_p, homogen_coord], dim=1)
    # v_homo: [num_points, 4, 4] x [num_points, 4, 1] --> [num_points, 4, 1]
    v_homo = torch.matmul(torch.inverse(T), torch.unsqueeze(pnts_p, dim=-1))
    # pnts: [num_points, 3]
    pnts = v_homo[:, :3, 0]

    return pnts


def vertices2joints(J_regressor, vertices):
    ''' Calculates the 3D joint locations from the vertices

    Parameters
    ----------
    J_regressor : torch.tensor JxV
        The regressor array that is used to calculate the joints from the
        position of the vertices
    vertices : torch.tensor BxVx3
        The tensor of mesh vertices

    Returns
    -------
    torch.tensor BxJx3
        The location of the joints
    '''

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    ''' Calculates the per vertex displacement due to the blend shapes


    Parameters
    ----------
    betas : torch.tensor Bx(num_betas)
        Blend shape coefficients
    shape_disps: torch.tensor Vx3x(num_betas)
        Blend shapes

    Returns
    -------
    torch.tensor BxVx3
        The per-vertex displacement due to shape deformation
    '''

    # Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l]
    # i.e. Multiply each shape displacement by its corresponding beta and
    # then sum them.
    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def batch_rodrigues(rot_vecs):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device, dtype = rot_vecs.device, rot_vecs.dtype

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def transform_mat(R, t):
    ''' Creates a batch of transformation matrices
        Args:
            - R: Bx3x3 array of a batch of rotation matrices
            - t: Bx3x1 array of a batch of translation vectors
        Returns:
            - T: Bx4x4 Transformation matrix
    '''
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]), F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents):
    """
    Applies a batch of rigid transformations to the joints

    Parameters
    ----------
    rot_mats : torch.tensor BxNx3x3
        Tensor of rotation matrices
    joints : torch.tensor BxNx3
        Locations of joints
    parents : torch.tensor BxN
        The kinematic tree of each object

    Returns
    -------
    posed_joints : torch.tensor BxNx3
        The locations of the joints after applying the pose rotations
    rel_transforms : torch.tensor BxNx4x4
        The relative (with respect to the root joint) rigid transformations
        for all the joints (path compressed transformation matrices)
    """

    joints = torch.unsqueeze(joints, dim=-1)

    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]]

    transforms_mat = transform_mat(rot_mats.view(-1, 3, 3),
                                   rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, 0]]
    for i in range(1, parents.shape[0]):  # From 1 to batch_size-1
        # Subtract the joint location at the rest pose
        # No need for rotation, since it's identity when at rest
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(torch.matmul(transforms, joints_homogen),
                                        [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms