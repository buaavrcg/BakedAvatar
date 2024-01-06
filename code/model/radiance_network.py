import torch
import torch.nn as nn

from model.embedder import FrequencyEmbedder
from utils.training_util import weights_init


class TextureBasisProjector(nn.Module):
    """
    Global condition projector (that runs per-frame on cpu) projects radiance conditions to
    dynamic weights of a tiny coefficient MLP (that runs in fragment shader) whose input is
    the spatially varying feature and output is the softmaxed coefficients of texture basis.

    P_global : (ψ, l) -> θ, where
        ψ∈R^{dim_expression} is the expression code,
        l∈R^{dim_frame_latent} is the per-frame latent code,
        θ∈R^{num_dynamic_parameters} is the weights of the tiny coefficient MLP.

    P_spatial : (f, n, v; θ) -> w, where
        f∈R^3 is the spatial positional feature,
        n∈R^3 is the normal in the deformed space,
        v∈R^3 is the view direction,
        w∈R^{num_texture_basis} is the coefficients of the texture basis.
    """
    def __init__(
        self,
        dim_pos_feature: int,
        dim_condition: int,
        num_texture_basis: int,
        dim_global_hidden=64,
        num_global_layers=3,
        dim_spatial_hidden=16,
        num_spatial_layers=2,
        init_type='default',
        temperature=1.0,
    ):
        super().__init__()
        self.dim_pos_feature = dim_pos_feature
        self.dim_condition = dim_condition
        self.dim_global_hidden = dim_global_hidden
        self.num_global_layers = num_global_layers
        self.dim_spatial_hidden = dim_spatial_hidden
        self.num_spatial_layers = num_spatial_layers
        self.temperature = temperature

        assert num_spatial_layers >= 1, "num_spatial_layers must be at least 1"
        self.spatial_mlp_dims = [dim_pos_feature + 6] \
                              + [dim_spatial_hidden] * (num_spatial_layers - 1) \
                              + [num_texture_basis]
        num_spatial_mlp_weights = [
            self.spatial_mlp_dims[i - 1] * self.spatial_mlp_dims[i]
            for i in range(1, len(self.spatial_mlp_dims))
        ]

        assert num_global_layers >= 1, "num_global_layers must be at least 1"
        dims = [dim_condition] + [dim_global_hidden] * (num_global_layers - 1) \
             + [sum(num_spatial_mlp_weights)]
        for i in range(self.num_global_layers):
            layer = nn.Linear(dims[i], dims[i + 1])
            layer.apply(weights_init(init_type))
            setattr(self, f'global_fc{i}', layer)
            if i < self.num_global_layers - 1:
                prelu = nn.PReLU(dims[i + 1], init=0.1)
                setattr(self, f'global_act{i}', prelu)

    def forward(self, pos_feature, normal, viewdir, *conds):
        x = torch.cat(conds, dim=-1)
        assert x.shape[-1] == self.dim_condition

        # Get spatial MLP weights from global MLP
        for i in range(self.num_global_layers):
            layer = getattr(self, f'global_fc{i}')
            x = layer(x)
            if i < self.num_global_layers - 1:
                act = getattr(self, f'global_act{i}')
                x = act(x)

        # Run spatial MLP on position features to get texture basis coefficients
        index = 0
        y = torch.cat([pos_feature, normal, viewdir], dim=-1)  # (N, dim_pos_feature + 6)
        for i in range(self.num_spatial_layers):
            dim_input = self.spatial_mlp_dims[i]
            dim_output = self.spatial_mlp_dims[i + 1]
            num_weights = dim_output * dim_input
            layer_weight = x[:, index:index + num_weights].view(-1, dim_output, dim_input)
            index += num_weights

            y = torch.matmul(layer_weight, y.unsqueeze(-1)).squeeze(-1)
            if i < self.num_spatial_layers - 1:
                y = nn.functional.relu(y, inplace=True)

        y = nn.functional.softmax(y * self.temperature, dim=-1)
        return y

    def export_weights(self):
        """
        Export weights of the global fully connected layer.
        Returns: a dict of layer name -> layer weight tensor
        """
        layers = {}

        for i in range(self.num_global_layers):
            fc = getattr(self, f'global_fc{i}')
            layers[f'fc{i}_weight'] = fc.weight.data
            layers[f'fc{i}_bias'] = fc.bias.data
            if i < self.num_global_layers - 1:
                act = getattr(self, f'global_act{i}')
                layers[f'act{i}_weight'] = act.weight.data

        return layers

    def load_weights(self, layers):
        """
        Load weights of the global fully connected layer.
        Args:
            layers: a dict of layer name -> layer weight tensor
        """
        for i in range(self.num_global_layers):
            fc = getattr(self, f'global_fc{i}')
            fc.weight.data = layers[f'fc{i}_weight'].to(fc.weight.device)
            fc.bias.data = layers[f'fc{i}_bias'].to(fc.bias.device)
            if i < self.num_global_layers - 1:
                act = getattr(self, f'global_act{i}')
                act.weight.data = layers[f'act{i}_weight'].to(act.weight.device)


class RadianceNetwork(nn.Module):
    """
    Linear blendable deferred radiance field in the canonical space.
    R : (x_c, θ_jaw, ψ, l) -> (f, alpha), where
        x_c∈R^3 is the canonical position,
        θ_jaw∈R is the jaw angle,
        ψ∈R^{dim_expression} is the expression code,
        l∈R^{dim_frame_latent} is the per-frame latent code,
        f∈R^{dim_radiance_feature} is the radiance feature.
        alpha∈R is the occupancy.
    """
    def __init__(
        self,
        dim_expression: int,
        dim_frame_latent: int,
        dim_radiance_feature: int,
        dim_pos_feature: int,
        num_texture_basis: int,
        pe_freqs=10,
        num_layers=8,
        num_base_layers=4,
        dim_hidden=128,
        skip_input_layers=[],
        init_type='default',
        texture_basis_args={},
        rawalpha_bias_init=0.0,
    ):
        super().__init__()
        self.pos_embedder = FrequencyEmbedder(num_freqs=pe_freqs)
        self.tex_projector = TextureBasisProjector(
            dim_pos_feature=dim_pos_feature,
            dim_condition=9 + dim_expression + dim_frame_latent,
            num_texture_basis=num_texture_basis,
            **texture_basis_args,
        )
        self.dim_radiance_feature = dim_radiance_feature
        self.num_texture_basis = num_texture_basis
        self.num_layers = num_layers
        self.num_base_layers = num_base_layers
        self.skip_input_layers = skip_input_layers
        assert num_layers > num_base_layers

        dims = [self.pos_embedder.dim_out] + [dim_hidden] * (num_layers - 1) \
             + [dim_radiance_feature]
        for i in range(self.num_layers):
            dim_in = dims[i] + (dims[0] if i in skip_input_layers else 0)
            if i < self.num_base_layers:
                layer = nn.Linear(dim_in, dims[i + 1])
                layer.apply(weights_init(init_type))
                setattr(self, f'l{i}', layer)
            else:
                for basis_idx in range(self.num_texture_basis):
                    layer = nn.Linear(dim_in, dims[i + 1])
                    layer.apply(weights_init(init_type))
                    if i == self.num_layers - 1:  # Initialize alpha weight
                        nn.init.constant_(layer.bias[-1], rawalpha_bias_init)
                    setattr(self, f'l{i}_basis{basis_idx}', layer)

        self.alpha_layer = nn.Sequential(
            nn.Linear(dims[self.num_base_layers], 1),
            nn.Sigmoid(),
        )
        self.alpha_layer.apply(weights_init(init_type))
        self.pos_feature_layer = nn.Sequential(
            nn.Linear(dims[self.num_base_layers], dim_hidden),
            nn.Softplus(),
            nn.Linear(dim_hidden, dim_pos_feature),
            nn.Sigmoid(),
        )
        self.pos_feature_layer.apply(weights_init(init_type))

    def forward(self, x, normal, viewdir, pose, exp, latent):
        """
        Args:
            x: (N, 3). Canonical positions.
            normal: (N, 3). Normal vectors.
            viewdir: (N, 3). View directions.
            pose: (N, 9). Jaw and eye pose angle.
            exp: (N, dim_expression). Expression code.
            latent: (N, dim_frame_latent). Per-frame latent code.
        Returns:
            feature: (N, dim_radiance_feature). Radiance feature.
            alpha: (N,). Occupancy in [0, 1].
            pos_feature: (N, dim_pos_feature). Spatially varying feature.
            texture_coef: (N, num_texture_basis). Normalized coefficients of the texture basis.
            features: (N, num_texture_basis, dim_radiance_feature). Radiance feature of texture basis.
        """
        pos_feature, fs, alpha = self.get_static_features(x)

        # project spatially varing feature and global conditions to texture coefficients
        texture_coef = self.tex_projector(
            pos_feature,
            normal,
            viewdir,
            pose,
            exp,
            latent,
        )  # (N, num_texture_basis)

        # blend textures with the given coefficients
        f = torch.sum(fs * texture_coef[..., None], dim=1)  # (N, dim_radiance_feature)

        return f, alpha, pos_feature, texture_coef, fs

    def get_static_features(self, x):
        """
        Get static features of the radiance field.
        Args:
            x: (N, 3). Canonical positions.
        Returns:
            pos_feature: (N, dim_pos_feature). Spatially varying feature.
            features: (N, num_texture_basis, dim_radiance_feature). Radiance feature of texture basis.
            alpha: (N,). Occupancy value in [0, 1].
        """
        input = self.pos_embedder(x)  # (N, dim_pos_embedder)

        x = input
        for i in range(self.num_base_layers):
            layer = getattr(self, f'l{i}')
            if i in self.skip_input_layers:
                x = torch.cat([x, input], dim=-1)
            x = layer(x)
            x = nn.functional.leaky_relu(x, inplace=True)

        features = []
        for basis_idx in range(self.num_texture_basis):
            y = x
            for i in range(self.num_base_layers, self.num_layers):
                layer = getattr(self, f'l{i}_basis{basis_idx}')
                if i in self.skip_input_layers:
                    y = torch.cat([y, input], dim=-1)
                y = layer(y)
                if i < self.num_layers - 1:
                    y = nn.functional.leaky_relu(y, inplace=True)
            features.append(y)
        features = torch.stack(features, dim=1)  # (N, num_texture_basis, dim_radiance_feature)
        features = torch.sigmoid(features)

        alpha = self.alpha_layer(x).squeeze(-1)  # (N, )
        alpha = torch.clamp(alpha * 1.002 - 0.001, 0.0, 1.0)
        pos_feature = self.pos_feature_layer(x)  # (N, dim_pos_feature)

        return pos_feature, features, alpha


class GridRadianceNetwork(nn.Module):
    """
    Linear blendable deferred radiance field in the canonical space.
    R : (x_c, θ_jaw, ψ, l) -> (f, alpha), where
        x_c∈R^3 is the canonical position,
        θ_jaw∈R is the jaw angle,
        ψ∈R^{dim_expression} is the expression code,
        l∈R^{dim_frame_latent} is the per-frame latent code,
        f∈R^{dim_radiance_feature} is the radiance feature.
        alpha∈R is the occupancy.
    """
    def __init__(
        self,
        dim_expression: int,
        dim_frame_latent: int,
        dim_radiance_feature: int,
        dim_pos_feature: int,
        num_texture_basis: int,
        bounding_box_radius=1.0,
        num_layers=3,
        dim_hidden=64,
        n_levels=12,
        n_features_per_level=2,
        base_resolution=16,
        per_level_scale=1.5,
        log2_hashmap_size=16,
        texture_basis_args={},
    ):
        super().__init__()
        import tinycudann as tcnn
        assert num_layers >= 2
        assert dim_hidden in [16, 32, 64, 128]
        self.tex_projector = TextureBasisProjector(
            dim_pos_feature=dim_pos_feature,
            dim_condition=9 + dim_expression + dim_frame_latent,
            num_texture_basis=num_texture_basis,
            **texture_basis_args,
        )
        self.bounding_box_radius = bounding_box_radius
        self.dim_radiance_feature = dim_radiance_feature
        self.dim_pos_feature = dim_pos_feature
        self.num_texture_basis = num_texture_basis

        self.ngp_network = tcnn.NetworkWithInputEncoding(
            n_input_dims=3,
            n_output_dims=dim_radiance_feature * self.num_texture_basis + 1 + dim_pos_feature,
            encoding_config={
                "otype":
                "Composite",
                "nested": [
                    {
                        "otype": "Identity"
                    },
                    {
                        "otype": "HashGrid",
                        "n_dims_to_encode": 3,
                        "n_levels": n_levels,
                        "n_features_per_level": n_features_per_level,
                        "log2_hashmap_size": log2_hashmap_size,
                        "base_resolution": base_resolution,
                        "per_level_scale": per_level_scale,
                    },
                ],
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": dim_hidden,
                "n_hidden_layers": num_layers - 1
            },
        )

    def forward(self, x, normal, viewdir, pose, exp, latent):
        """
        Args:
            x: (N, 3). Canonical positions.
            normal: (N, 3). Normal vectors.
            pose: (N, 3). Jaw pose angle.
            exp: (N, dim_expression). Expression code.
            latent: (N, dim_frame_latent). Per-frame latent code.
        Returns:
            feature: (N, dim_radiance_feature). Radiance feature.
            alpha: (N,). Occupancy in [0, 1].
            pos_feature: (N, dim_pos_feature). Spatially varying feature.
            texture_coef: (N, num_texture_basis). Normalized coefficients of the texture basis.
            features: (N, num_texture_basis, dim_radiance_feature). Radiance feature of texture basis.
            alphas: (N, num_texture_basis). Raw alpha of texture basis before applying sigmoid.
        """
        pos_feature, fs, alpha = self.get_static_features(x)

        # project spatially varing feature and global conditions to texture coefficients
        texture_coef = self.tex_projector(
            pos_feature,
            normal,
            viewdir,
            pose,
            exp,
            latent,
        )  # (N, num_texture_basis)

        # blend textures with the given coefficients
        f = torch.sum(fs * texture_coef[..., None], dim=1)  # (N, dim_radiance_feature)

        return f, alpha, pos_feature, texture_coef, fs

    def get_static_features(self, x):
        """
        Get static features of the radiance field.
        Args:
            x: (N, 3). Canonical positions.
        Returns:
            pos_feature: (N, dim_pos_feature). Spatially varying feature.
            features: (N, num_texture_basis, dim_radiance_feature). Radiance feature of texture basis.
            alpha: (N,). Alpha of texture basis before applying sigmoid.
        """
        # normalize coordinates to [0, 1] according to bounding aabb radius
        x_normed = (x + self.bounding_box_radius) / (2 * self.bounding_box_radius)
        # get feature from grid encoding
        x = self.ngp_network(x_normed).float()  # (N, n_output_dims)

        # get radiance feature, raw alphas, and positional feature
        f_dim = self.num_texture_basis * self.dim_radiance_feature
        features = x[:, :f_dim].view(-1, self.num_texture_basis, self.dim_radiance_feature)
        alpha = x[:, f_dim:f_dim + 1]  # (N, 1)
        pos_feature = x[:, f_dim + 1:]  # (N, dim_pos_feature)

        # clamp features and alphas to range [0, 1]
        features = torch.sigmoid(features)
        alpha = torch.sigmoid(alpha.squeeze(-1))
        alpha = torch.clamp(alpha * 1.002 - 0.001, 0.0, 1.0)

        return pos_feature, features, alpha


class RadianceDecoder(nn.Module):
    def __init__(
        self,
        dim_radiance_feature: int,
        dir_embedding_freqs=0,
        dim_rgb=3,
        num_layers=2,
        dim_hidden=16,
        init_type='default',
    ):
        super().__init__()
        self.dir_embedder = FrequencyEmbedder(num_freqs=dir_embedding_freqs)
        self.num_layers = num_layers

        dims = [dim_radiance_feature + self.dir_embedder.dim_out] \
             + [dim_hidden] * (num_layers - 1) + [dim_rgb]
        for i in range(self.num_layers):
            layer = nn.Linear(dims[i], dims[i + 1])
            layer.apply(weights_init(init_type))
            setattr(self, f'fc{i}', layer)

    def forward(self, radiance_feature: torch.Tensor, viewdir: torch.Tensor) -> torch.Tensor:
        """
        Args:
            radiance_feature: (N, dim_radiance_feature). Accumulated radiance feature along rays.
            viewdir: (N, 3). View direction in unit vector.
        Returns:
            rgb: (N, 3). RGB color in range [0, 1].
        """
        dir_embedding = self.dir_embedder(viewdir)
        x = torch.cat([radiance_feature, dir_embedding], -1)

        for i in range(self.num_layers):
            layer = getattr(self, f'fc{i}')
            x = layer(x)
            if i < self.num_layers - 1:
                x = nn.functional.relu(x, inplace=True)

        x = torch.sigmoid(x)
        return x
