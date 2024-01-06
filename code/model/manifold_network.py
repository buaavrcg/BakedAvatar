import math
import torch
import torch.nn as nn

from model.embedder import FrequencyEmbedder
from utils.training_util import weights_init


def geometry_init(layer_idx, total_layers, skip_input_layers, input_dim=3, radius=0.0):
    def init_fn(m):
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            nn.init.normal_(m.weight, 0, math.sqrt(2 / num_output))
            if layer_idx in skip_input_layers:
                nn.init.constant_(m.weight[:, -input_dim:], 0)
            nn.init.constant_(m.bias, 0)

    def init_fn_first_layer(m):
        if isinstance(m, nn.Linear):
            num_output = m.weight.size(0)
            nn.init.normal_(m.weight[:, :3], 0, math.sqrt(2 / num_output))
            nn.init.constant_(m.weight[:, 3:], 0.0)
            nn.init.constant_(m.bias, 0.0)

    def init_fn_last_layer(m):
        if isinstance(m, nn.Linear):
            num_input = m.weight.size(-1)
            nn.init.constant_(m.weight, math.sqrt(math.pi / num_input))
            nn.init.constant_(m.bias, -radius)

    return init_fn_last_layer if layer_idx == total_layers - 1 else \
           init_fn_first_layer if layer_idx == 0 else init_fn


class ManifoldNetwork(nn.Module):
    """
    Manifold field in the canonical space.
    M : (x_c) -> s, x_c∈R^3 is the canonical position, s∈R is the scalar level.
    """
    def __init__(self,
                 pe_freqs=0,
                 num_layers=3,
                 dim_hidden=128,
                 init_type='geometry',
                 init_radius=1.0,
                 freq_decay=1.0,
                 skip_input_layers=[],
                 weight_norm=True):
        super().__init__()
        self.pos_embedder = FrequencyEmbedder(num_freqs=pe_freqs)
        self.num_layers = num_layers
        self.skip_input_layers = skip_input_layers

        dims = [self.pos_embedder.dim_out] + [dim_hidden] * (num_layers - 1) + [1]
        for i in range(self.num_layers):
            dim_in = dims[i] + (dims[0] if i in skip_input_layers else 0)
            layer = nn.Linear(dim_in, dims[i + 1])

            if init_type == 'geometry':
                layer.apply(
                    geometry_init(i, self.num_layers, skip_input_layers, dims[0], init_radius))
            else:
                layer.apply(weights_init(init_type))

            if weight_norm:
                layer = nn.utils.weight_norm(layer)

            setattr(self, f'l{i}', layer)

        self.activation = nn.Softplus(beta=100)
        freq_weight = torch.cat([
            torch.ones(3),
            torch.pow(freq_decay, (1.0 + torch.arange(pe_freqs))).repeat_interleave(6)
        ])
        self.register_buffer('freq_weight', freq_weight.unsqueeze(0), False)

    def forward(self, x):
        """
        Args:
            x: (N, 3). Canonical position.
        Returns:
            s: (N,). Scalar level ∈ R.
        """
        x = input = self.pos_embedder(x) * self.freq_weight

        for i in range(self.num_layers):
            layer = getattr(self, f'l{i}')
            if i in self.skip_input_layers:
                x = torch.cat([x, input], dim=-1)
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)

        return x


class GridManifoldNetwork(nn.Module):
    """
    Manifold field in the canonical space, backed by NGP.
    M : (x_c) -> s, x_c∈R^3 is the canonical position, s∈R is the scalar level.
    """
    def __init__(self,
                 bounding_box_radius=1.0,
                 num_layers=2,
                 dim_hidden=64,
                 n_levels=6,
                 n_features_per_level=2,
                 log2_hashmap_size=19,
                 base_resolution=16,
                 per_level_scale=2.0,
                 init_type='geometry',
                 init_radius=1.0):
        super().__init__()
        import tinycudann as tcnn
        self.bounding_box_radius = bounding_box_radius
        self.num_layers = num_layers
        self.dim_encoding = n_levels * n_features_per_level
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale,
            },
            dtype=torch.float32,
        )

        dims = [3 + self.dim_encoding] + [dim_hidden] * (num_layers - 1) + [1]
        for i in range(self.num_layers):
            layer = nn.Linear(dims[i], dims[i + 1])
            if init_type == 'geometry':
                layer.apply(geometry_init(i, self.num_layers, init_radius))
            else:
                layer.apply(weights_init(init_type))
            setattr(self, f'l{i}', layer)

        self.activation = nn.ReLU(inplace=True)  # nn.Softplus(beta=100)

    def forward(self, x):
        # normalize coordinates to [0, 1] according to bounding aabb radius
        x_normed = (x + self.bounding_box_radius) / (2 * self.bounding_box_radius)
        # get feature from grid encoding
        f = self.encoding(x_normed)  # (N, dim_encoding)
        # concatenate raw coordinates with grid features
        x = torch.cat([x, f], dim=-1)  # (N, 3 + dim_encoding)

        for i in range(self.num_layers):
            layer = getattr(self, f'l{i}')
            x = layer(x)
            if i < self.num_layers - 1:
                x = self.activation(x)

        return x
