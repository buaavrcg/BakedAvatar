import math
import torch
import torch.nn as nn

from model.embedder import FrequencyEmbedder
from model.root_solver import broyden
from utils.training_util import weights_init


def batch_jacobian(y: torch.Tensor, x: torch.Tensor, create_graph=False, retain_graph=True):
    """
    Compute the Jacobian matrix of y=f(x) w.r.t. x.

    Args:
        y: (N, M), N is the batch size, M is the output dimension.
        x: (N, K), K is the input dimension.

    Returns:
        J: (N, M, K), the Jacobian matrix of y w.r.t. x.
    """

    jacobian = []
    for i in range(y.shape[1]):
        d_out = torch.zeros_like(y, requires_grad=False)
        d_out[:, i] = 1
        grad, = torch.autograd.grad(y,
                                    x,
                                    grad_outputs=d_out,
                                    create_graph=create_graph,
                                    retain_graph=i < y.shape[1] - 1 or retain_graph)
        jacobian.append(grad)  # (N, K)

    return torch.stack(jacobian, dim=1)  # (N, M, K)


def geometry_init(layer_idx, total_layers):
    def init_fn(m):
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            nn.init.normal_(m.weight, 0.0, math.sqrt(2 / num_output))
            nn.init.constant_(m.bias, 0.0)

    def init_fn_first_layer(m):
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            nn.init.normal_(m.weight[:, :3], 0, math.sqrt(2 / num_output))
            nn.init.constant_(m.weight[:, 3:], 0.0)
            nn.init.constant_(m.bias, 0.0)

    def init_fn_last_layer(m):
        # initialize blendshapes direction to be very close to zero, and initialize
        # skinning weights to be equal for every bone after softmax activation
        if isinstance(m, nn.Linear):
            nn.init.constant_(m.weight, 0.0)
            nn.init.constant_(m.bias, 0.0)

    return init_fn_last_layer if layer_idx == total_layers - 1 else \
           init_fn_first_layer if layer_idx == 0 else init_fn


class ForwardDeformer(nn.Module):
    """
    Forward deformer explicitly converts points from canonical space to deformed space
    using FLAME deformation, and it also implicitly converts points from deformed space
    to canonical space with iterative correspondence search.
    
    Forward deformation can be denoted as F_fwd : (x_c, θ, ψ) -> x_d, where
        x_c∈R^3 is the canonical position,
        θ∈3(K+1) is the joint poses with K bones and the root,
        ψ∈R^{dim_expression} is the expression code,
        x_d∈R^3 is the deformed position.

    For each canonical point x_c, we compute deformed point following the FLAME formulation,
    F_fwd(x_c, θ, ψ) = LBS(x_c + B_E(ψ;E) + B_P(θ;P), J(x_c + B_E(ψ;E)), θ, W), where
        LBS is the linear blend skinning function,
        B_E and B_P is the expression and bone pose blend shape function,
        E and P are the expression and bone pose blend shape direction at x_c,
        J is the joint function, W is the bone weights at x_c.
    Pos-varying P, E, W is predicted by the deformation network: D : (x_c) -> P, E, W.

    Code is adopted based on IMavatar: https://github.com/zhengyuf/IMavatar.
    """
    def __init__(
        self,
        flame: nn.Module,
        dim_expression: int,
        pe_freqs=0,
        num_layers=4,
        dim_hidden=128,
        init_type='geometry',
        ghostbone=False,
        bone_weight_temp=20.0,
        weight_norm=True,
    ):
        super().__init__()
        self.pos_embedder = FrequencyEmbedder(num_freqs=pe_freqs)
        self.get_flame = lambda: flame
        self.dim_expression = dim_expression
        self.num_layers = num_layers
        self.init_bones = [0, 2, 3] if ghostbone else [0, 1, 2]
        self.num_bones = 6 if ghostbone else 5
        self.bone_weight_temp = bone_weight_temp

        # output dim consists of pose dirs, expression dirs and skinning weights
        dims = [self.pos_embedder.dim_out] + [dim_hidden] * (num_layers - 1)
        for i in range(self.num_layers - 1):
            layer = nn.Linear(dims[i], dims[i + 1])

            if init_type == 'geometry':
                layer.apply(geometry_init(i, self.num_layers))
            else:
                layer.apply(weights_init(init_type))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            setattr(self, f'l{i}', layer)

        self.activation = nn.Softplus(beta=100)
        self.blendshapes = nn.Linear(dims[-1], 3 * dim_expression + 36 * 3)
        self.blendshapes.apply(geometry_init(self.num_layers - 1, self.num_layers))
        self.skinning_layer = nn.Linear(dims[-1], dims[-1])
        self.skinning = nn.Linear(dims[-1], self.num_bones)
        self.skinning_layer.apply(geometry_init(self.num_layers - 1, self.num_layers + 1))
        self.skinning.apply(geometry_init(self.num_layers, self.num_layers + 1))
        if weight_norm:
            self.skinning_layer = nn.utils.weight_norm(self.skinning_layer)

    def query_weights(self, x):
        """
        Query pose dirs, expression dirs and skinning weights at canonical point x.

        Args:
            x: (N, 3). Points in canonical space.
        Returns:
            E: (N, 3, dim_expression). Expression blend shape directions
            P: (N, 36, 3). Pose blend shape directions
            W: (N, num_bones). Skinning weights
        """
        x = self.pos_embedder(x)

        for i in range(self.num_layers - 1):
            layer = getattr(self, f'l{i}')
            x = self.activation(layer(x))

        # split output into pose dirs, expression dirs and skinning weights
        d_e = self.dim_expression * 3
        d_p = d_e + 36 * 3
        blendshapes = self.blendshapes(x)
        E = blendshapes[:, :d_e].view(-1, 3, self.dim_expression)
        P = blendshapes[:, d_e:d_p].view(-1, 36, 3)
        
        W = self.skinning(self.activation(self.skinning_layer(x)))
        # use a high temperature to make the distribution more discrete
        W = torch.softmax(self.bone_weight_temp * W, dim=1)

        return E, P, W

    def forward(self, x_c, betas, transforms, pose_feature):
        """
        Transform canonical points to deformed points using FLAME deformation.
        Args:
            x_c: (N, 3). Points in canonical space.
            betas: (N, dim_expression). Expression parameters.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
            pose_feature: (N, 36). Pose coefficients for pose-related blendshapes.
        Returns:
            x_d: (N, 3) Deformed correspondences of canonical points.
        """
        E, P, W = self.query_weights(x_c)
        x_d = self.get_flame().forward_points(x_c, betas, transforms, pose_feature, E, P, W)
        return x_d

    def init_canonical_points(self, x_d, transforms):
        """
        Get initial canonical points by inverse rigid transformation.
        Args:
            x_d: (N, 3). Points in the deformed space.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
        Returns:
            x_c_init: (N, num_init_bones, 3). Initial canonical points for each bone.
        """
        x_c_inits = []

        for i in self.init_bones:
            W = torch.zeros(transforms.shape[:2], device=x_d.device)
            W[:, i] = 1.0
            x_c_init = self.get_flame().inverse_skinning_points(x_d, transforms, W)
            x_c_inits.append(x_c_init)
            
        return torch.stack(x_c_inits, dim=1)

    def search_canonical_points(self, x_d, x_c_init, betas, transforms, pose_feature):
        """
        Search for the canonical points that correspond to the deformed points.
        Args:
            x_d: (N, 3). Points in the deformed space.
            x_c_init: (N, num_init_bones, 3). Initial canonical points for each bone.
            betas: (N, dim_expression). Expression parameters.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
            pose_feature: (N, 36). Pose coefficients for pose-related blendshapes.
        Returns:
            x_c: (N, num_init_bones, 3). Searched canonical correspondences of x_d.
            valid_mask: (N, num_init_bones). Valid mask of converged canonical points.
        """
        N, num_init_bones, _ = x_c_init.shape
        x_c_init = x_c_init.view(N * num_init_bones, 3)
        # reshape to (B * num_init_bones, D)
        x_d_target = x_d.repeat_interleave(num_init_bones, dim=0)
        betas = betas.repeat_interleave(num_init_bones, dim=0)
        transforms = transforms.repeat_interleave(num_init_bones, dim=0)
        pose_feature = pose_feature.repeat_interleave(num_init_bones, dim=0)

        # compute initial approximated jacobians using inverse-skinning
        _, _, W = self.query_weights(x_c_init)
        # n: num_points, j: num_joints, k: 16 (4x4)
        J = torch.einsum('nj, njk -> nk', W, transforms.view(*W.shape, 16))
        J_inv_init = J.view(-1, 4, 4)[:, :3, :3].inverse()  # (N, 3, 3)

        # error function for root finding
        def error_f(x_c_opt, mask=None):
            x_c_opt = x_c_opt.reshape(N * num_init_bones, 3)
            x_d = self(x_c_opt[mask] if mask is not None else x_c_opt,
                       betas[mask] if mask is not None else betas,
                       transforms[mask] if mask is not None else transforms,
                       pose_feature[mask] if mask is not None else pose_feature)
            error = x_d - x_d_target[mask]
            return error.unsqueeze(-1)  # reshape error to (*, D, 1) for broyden solver

        x_c_init = x_c_init.unsqueeze(-1)  # reshape init to (*, D, 1) for broyden solver
        with torch.no_grad():
            x_c_opt, _, valid_mask = broyden(error_f, x_c_init, J_inv_init)

        x_c_opt = x_c_opt.reshape(N, num_init_bones, 3)
        valid_mask = valid_mask.reshape(N, num_init_bones)

        return x_c_opt, valid_mask

    def forward_inverse(self, x_d, betas, transforms, pose_feature):
        """
        Find (multiple) canonical correspondences of deformed points.
        Args:
            x_d: (N, 3). Points in the deformed space.
            betas: (N, dim_expression). Expression parameters.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
            pose_feature: (N, 36). Pose coefficients for pose-related blendshapes.
        Returns:
            x_c: (N, num_init_bones, 3). Searched canonical correspondences of x_d.
            valid_mask: (N, num_init_bones). Valid mask of converged canonical points.
        """
        x_c_init = self.init_canonical_points(x_d, transforms)
        x_c_opt, valid_mask = self.search_canonical_points(x_d, x_c_init, betas, transforms,
                                                           pose_feature)

        return x_c_opt, valid_mask

    @torch.enable_grad()
    def get_differential_canonical_points(self,
                                          x_c,
                                          betas,
                                          transforms,
                                          pose_feature,
                                          x_d_input=None):
        """
        Get a differential version of x_c with gradients w.r.t. the network parameters.
        Args:
            x_c: (N, 3). Points in canonical space.
            betas: (N, dim_expression). Expression parameters.
            transforms: (N, num_joints, 4, 4). Transformation matrices of each joint.
            pose_feature: (N, 36). Pose coefficients for pose-related blendshapes.
            x_d_input: (N, 3). Optional original input points in deformed space, can be
                used for back-propogating gradients for optimizing camera extrinsic.
        Returns:
            x_c_with_grad: (N, 3). Points in canonical space with gradients w.r.t. net params.
        """
        if x_c.shape[0] == 0:
            return x_c.detach()

        x_c = x_c.requires_grad_(True)
        x_d = self.forward(x_c, betas, transforms, pose_feature)

        constant = (x_d - x_d_input) if x_d_input is not None else x_d
        grad_inv = batch_jacobian(constant, x_c).inverse()
        correction = constant - constant.detach()  # value to zero while keeping gradients
        correction = -torch.einsum("nij,nj->ni", grad_inv, correction)

        # adding implicit diff to autodiff: x_c = x_c_opt + 0 and x_c' = correction'
        x_c = x_c.detach() + correction

        return x_c
        