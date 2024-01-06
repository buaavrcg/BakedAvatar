import numpy as np
import torch
import torch.nn as nn
import nvdiffrast.torch as dr

from flame.FLAME import FLAME
from model.radiance_network import TextureBasisProjector
from utils.render_util import get_projection_and_view, alpha_integration
from utils.training_util import fake_quant


class RasterizationModel(nn.Module):
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
        optimize_vertices=False,
        optimize_normals=False,
        optimize_textures=True,
        optimize_blendshapes=False,
        optimize_lbs_weights=False,
        rasterize_flame_weights=False,
        same_alpha_across_basis=True,
        force_inner_alpha_full=True,
        enable_texture_qat=True,
        **kwargs,
    ):
        super().__init__()
        self.dim_expression = dim_expression
        self.dim_frame_latent = dim_frame_latent
        self.dim_radiance_feature = dim_radiance_feature
        self.dim_pos_feature = dim_pos_feature
        self.num_texture_basis = num_texture_basis
        self.ghostbone = ghostbone
        self.flame = FLAME(100, dim_expression, shape_params, canonical_exp, ghostbone)
        self.far_distance = 1e2
        self.optimize_vertices = optimize_vertices
        self.optimize_normals = optimize_normals
        self.optimize_textures = optimize_textures
        self.optimize_blendshapes = optimize_blendshapes
        self.optimize_lbs_weights = optimize_lbs_weights
        self.rasterize_flame_weights = rasterize_flame_weights
        self.same_alpha_across_basis = same_alpha_across_basis
        self.force_inner_alpha_full = force_inner_alpha_full
        self.enable_texture_qat = enable_texture_qat
        self.glctx = dr.RasterizeCudaContext()

        try:
            tex_projector_args = kwargs["radiance_network_args"]["texture_basis_args"]
        except:
            tex_projector_args = {}
        self.tex_projector = TextureBasisProjector(
            dim_pos_feature=dim_pos_feature,
            dim_condition=9 + dim_expression + dim_frame_latent,
            num_texture_basis=num_texture_basis,
            **tex_projector_args,
        )
        if not optimize_textures:
            for p in self.tex_projector.parameters():
                p.requires_grad = False

    def load_mesh_data(self, mesh_data_dict) -> None:
        """Load mesh data from the given dict."""
        self.mesh_data = mesh_data_dict

        self.vertices = nn.ParameterList()  # num_meshes * (V, 3)
        self.faces = nn.ParameterList()  # num_meshes * (F, 3)
        self.shapedirs = nn.ParameterList()  # num_meshes * (V, 50, 3)
        self.posedirs = nn.ParameterList()  # num_meshes * (V, 36, 3)
        self.lbs_weights = nn.ParameterList()  # num_meshes * (V, 5 or 6)
        self.normals = nn.ParameterList()  # num_meshes * (V, 3)
        self.uvs = nn.ParameterList()  # num_meshes * (UV, 2)
        self.faces_uv = nn.ParameterList()  # num_meshes * (FUV, 3)
        self.position_texture = nn.ParameterList()  # num_meshes * (H, W, dim_position_feature)
        self.radiance_textures = nn.ParameterList(
        )  # num_meshes * (num_texture_basis, H, W, dim_radiance_feature+1)
        self.radiance_textures_alpha = nn.ParameterList()  # num_meshes * (1, H, W, 1)

        # load mesh data to nn.Parameter
        for mesh in self.mesh_data['meshes']:
            vertices = torch.from_numpy(mesh['vertices'])
            faces = torch.from_numpy(mesh['faces']).int()
            shapedirs = torch.from_numpy(mesh['shapedirs']).permute(0, 2, 1)
            posedirs = torch.from_numpy(mesh['posedirs'])
            lbs_weights = torch.from_numpy(mesh['lbs_weights'])
            normals = torch.from_numpy(mesh['normals'])
            uvs = torch.from_numpy(mesh['uvs'])
            faces_uv = torch.from_numpy(mesh['faces_uv']).int()
            pos_tex = torch.from_numpy(mesh['position_texture'].astype(np.float32) / 255.0)
            rad_tex = torch.from_numpy(mesh['radiance_textures'].astype(np.float32) / 255.0)

            self.vertices.append(nn.Parameter(vertices, self.optimize_vertices))
            self.faces.append(nn.Parameter(faces, False))
            self.shapedirs.append(nn.Parameter(shapedirs, self.optimize_blendshapes))
            self.posedirs.append(nn.Parameter(posedirs, self.optimize_blendshapes))
            self.lbs_weights.append(nn.Parameter(lbs_weights, self.optimize_lbs_weights))
            self.normals.append(nn.Parameter(normals, self.optimize_normals))
            self.uvs.append(nn.Parameter(uvs, False))
            self.faces_uv.append(nn.Parameter(faces_uv, False))
            self.position_texture.append(nn.Parameter(pos_tex, self.optimize_textures))
            self.radiance_textures.append(nn.Parameter(rad_tex, self.optimize_textures))
            if self.same_alpha_across_basis:
                rad_tex_alpha = torch.mean(rad_tex[..., -1:], dim=0, keepdim=True)
                self.radiance_textures_alpha.append(
                    nn.Parameter(rad_tex_alpha, self.optimize_textures))

        # load tex projector weights
        weights_dict = {}
        for k, weights in self.mesh_data['global_mlp'].items():
            weights_dict[k] = torch.from_numpy(weights).float()
        self.tex_projector.load_weights(weights_dict)

    def export_mesh_data(self) -> dict:
        """Export the mesh data as a dict."""
        # update mesh data from nn.Parameter
        for i, mesh in enumerate(self.mesh_data['meshes']):
            if self.optimize_vertices:
                mesh['vertices'] = self.vertices[i].detach().cpu().numpy()
            if self.optimize_normals:
                mesh['normals'] = self.normals[i].detach().cpu().numpy()
            if self.optimize_textures:
                pos_tex = self.position_texture[i].detach().cpu().numpy()
                rad_tex = self.radiance_textures[i].detach().cpu().numpy()
                if self.same_alpha_across_basis:
                    rad_tex[..., -1:] = self.radiance_textures_alpha[i].detach().cpu().numpy()
                if self.force_inner_alpha_full and i == 0:
                    rad_tex[:, -1] = 1.0
                pos_tex = np.clip(np.around(pos_tex * 255), 0, 255)
                rad_tex = np.clip(np.around(rad_tex * 255), 0, 255)
                mesh['position_texture'] = pos_tex.astype(np.uint8)
                mesh['radiance_textures'] = rad_tex.astype(np.uint8)
            if self.optimize_blendshapes:
                mesh['shapedirs'] = self.shapedirs[i].detach().permute(0, 2, 1).cpu().numpy()
                mesh['posedirs'] = self.posedirs[i].detach().cpu().numpy()
            if self.optimize_lbs_weights:
                mesh['lbs_weights'] = self.lbs_weights[i].detach().cpu().numpy()

        # update tex projector weights
        if self.optimize_textures:
            weights_dict = self.tex_projector.export_weights()
            for k, weights in weights_dict.items():
                weights_dict[k] = weights.detach().cpu().numpy().astype(np.float32)
            self.mesh_data['global_mlp'] = weights_dict

        return self.mesh_data

    def transform_pos(self, pos_world, normals_world, intrinsic, extrinsic, img_res):
        """
        Transform world space positions and normals to view and clip space.
        Args:
            pos_world: (batch_size, num_vertices, 3). Positions in world space.
            normals_world: (batch_size, num_vertices, 3). Normals in world space.
            intrinsic: (batch_size, 3, 3). Intrinsic camera parameters.
            extrinsic: (batch_size, 3, 4). Extrinsic camera parameters.
            img_res: (batch_size, 2). Image resolution.
        Returns:
            pos_clip: (batch_size, num_vertices, 4). Positions in clip space.
            pos_view: (batch_size, num_vertices, 3). Positions in view space.
            normals_view: (batch_size, num_vertices, 3). Normals in view space.
        """
        P, V = get_projection_and_view(intrinsic, extrinsic, img_res)
        VP = torch.matmul(P, V)
        V_normal = torch.inverse(V).transpose(1, 2)

        # to homogeneous_coords: (x, y, z) -> (x, y, z, 1)
        pos_world = torch.cat([pos_world, torch.ones_like(pos_world[..., :1])], dim=-1)

        pos_clip = torch.matmul(pos_world, VP.permute(0, 2, 1))
        pos_view = torch.matmul(pos_world, V.permute(0, 2, 1))[:, :, :3]
        normals_view = torch.matmul(normals_world, V_normal[:, :3, :3].permute(0, 2, 1))

        return pos_clip, pos_view, normals_view

    def render(self, pos, attrs, uvs, tri, tri_uv, tex, img_res):
        """
        Differential rendering the mesh and returns rendered frame buffer.
        Args:
            pos: (batch_size, num_vertices, 4). Vertex positions in clip space.
            attrs: (batch_size, num_vertices, num_attributes). Attributes of each vertex.
            uvs: (num_uvs, 2). UV coordinates of each vertex.
            tri: (num_faces, 3). Vertex indices of each triangle.
            tri_uv: (num_faces, 3). UV indices of each triangle.
            tex: (batch_size, tex_height, tex_width, num_channels). Texture to render.
            img_res: (batch_size, 2). Resolution of the rendered frame buffer.
        Returns:
            attr_buffer: (batch_size, H, W, num_attributes). Rendered attribute frame buffer.
            tex_buffer: (batch_size, H, W, num_channels). Rendered texture frame buffer.
            mask: (batch_size, H, W). Mask of whether the pixel contains triangle.
        """
        B, V = pos.shape[0], uvs.shape[0]
        new_pos = torch.empty(B, V, pos.shape[-1], dtype=pos.dtype, device=pos.device)
        new_attrs = torch.empty(B, V, attrs.shape[-1], dtype=attrs.dtype, device=attrs.device)
        tri_uv_long = tri_uv.long()
        tri_long = tri.long()
        for i in range(B):
            new_pos[i, tri_uv_long] = pos[i, tri_long]
            new_attrs[i, tri_uv_long] = attrs[i, tri_long]
        pos = new_pos.contiguous()
        attrs = new_attrs.contiguous()
        uvs = uvs.contiguous()
        tri = tri_uv.contiguous()
        tex = tex.contiguous()

        # rasterize triangles
        assert torch.all(img_res == img_res[0]), "img_res should be the same for all samples."
        rast_out, _ = dr.rasterize(self.glctx, pos, tri, img_res[0].cpu().tolist())

        # interpolate attributes
        attr_buffer, _ = dr.interpolate(attrs, rast_out, tri)
        texcoord, _ = dr.interpolate(uvs, rast_out, tri)

        # sample textures
        tex_buffer = dr.texture(tex.unsqueeze(0), texcoord, filter_mode='linear')

        # anti-aliasing
        tex_buffer = dr.antialias(tex_buffer, rast_out, pos, tri)

        return attr_buffer, tex_buffer, rast_out[..., -1] > 0

    def save_mesh(self, save_paths, v_posed, n_posed, layer_idx):
        """
        Save posed mesh to the given paths.
        Args:
            save_paths: list of path to save the mesh obj file.
            v_posed: (batch_size, num_vertices, 3). Posed vertex positions.
            n_posed: (batch_size, num_vertices, 3). Pose normal directions.
            layer_idx: int. The layer index.
        """
        import os
        from pathlib import Path
        from pytorch3d.io import save_obj
        for i in range(v_posed.shape[0]):
            save_dir = os.path.dirname(save_paths[i])
            save_name, save_ext = os.path.splitext(os.path.basename(save_paths[i]))
            save_obj(
                Path(os.path.join(save_dir, f"{save_name}_layer{layer_idx}{save_ext}")),
                verts=v_posed[i],
                faces=self.faces[layer_idx].data,
            )

    def forward(self, inputs):
        """
        Args are in a dict of inputs:
            img_res: (batch_size, 2). Image resolution.
            intrinsic: (batch_size, 3, 3). Camera intrinsic matrix.
            extrinsic: (batch_size, 4, 4). Camera extrinsic matrix.
            expression: (batch_size, dim_expression). Expression parameters in FLAME.
            pose: (batch_size, 15). Pose parameters in FLAME.
            background_rgb: (batch_size, 3). Background color.
        Returns a dictionary of outputs:
            rgb: (batch_size, num_rays, 3). Final rgb output.
            alpha: (batch_size, num_rays). Final alpha output.
            depth: (batch_size, num_rays). Alpha weighted average of depth.
            mask: (batch_size, num_rays). Predicted mask of the object.
            mask_manifold: (batch_size, num_rays, num_levels). Valid mask of manifold levels.
            alpha_manifold: (batch_size, num_rays, num_levels). Alpha of manifold levels.
            normal: (batch_size, num_rays, num_levels, 3). Normal of manifold levels.
            dist: (batch_size, num_rays, num_levels). Intersection distance in deformed space.
        """
        intrinsic = inputs['intrinsic']  # (batch_size, 3, 3)
        extrinsic = inputs['extrinsic']  # (batch_size, 4, 4)
        expression = inputs['expression']  # (batch_size, dim_expression)
        pose = inputs['pose']  # (batch_size, 15)
        batch_size, device = expression.shape[0], expression.device
        frame_latent = torch.zeros(batch_size, self.dim_frame_latent, device=device)

        # get FLAME pose feature and transforms from expression and pose parameters
        # pose_feature: (batch_size, 36); transforms: (batch_size, num_joints, 4, 4)
        _, pose_feature, transforms = self.flame(expression, pose)

        # differential rasterization for layered meshes (draw from outer layer to inner layer)
        features = []
        alphas = []
        dists = []
        masks = []
        normals = []
        if self.rasterize_flame_weights:
            shapedirs_list = []
            posedirs_list = []
            lbs_weights_list = []
        texture_coefs = []
        attr_dims = np.cumsum([3, 3, 3 * self.dim_expression, 36 * 3])
        for i in reversed(range(len(self.vertices))):
            num_vertices = len(self.vertices[i])
            # get posed world space vertices and normals
            # shapedirs: (batch_size, num_vertices, 3, dim_expression)
            # posedirs: (batch_size, num_vertices, 36, 3)
            # lbs_weights: (batch_size, num_vertices, num_bones)
            shapedirs = self.shapedirs[i].unsqueeze(0).expand(batch_size, -1, -1, -1)
            posedirs = self.posedirs[i].unsqueeze(0).expand(batch_size, -1, -1, -1)
            lbs_weights = self.lbs_weights[i].unsqueeze(0).expand(batch_size, -1, -1)
            v_posed, n_posed = self.flame.forward_original_points(
                self.vertices[i].unsqueeze(0).expand(batch_size, -1, -1).flatten(0, 1),
                self.normals[i].unsqueeze(0).expand(batch_size, -1, -1).flatten(0, 1),
                expression.unsqueeze(1).expand(-1, num_vertices, -1).flatten(0, 1),
                transforms.unsqueeze(1).expand(-1, num_vertices, -1, -1, -1).flatten(0, 1),
                pose_feature.unsqueeze(1).expand(-1, num_vertices, -1).flatten(0, 1),
                shapedirs.flatten(0, 1),
                posedirs.flatten(0, 1),
                lbs_weights.flatten(0, 1),
            )
            v_posed = v_posed.reshape(batch_size, num_vertices, 3)
            n_posed = n_posed.reshape(batch_size, num_vertices, 3)

            if 'mesh_save_path' in inputs:
                self.save_mesh(inputs['mesh_save_path'], v_posed, n_posed, i)

            # transform vertices to view+clip space, normals to view space
            # pos_clip: (batch_size, num_vertices, 4)
            # pos_view: (batch_size, num_vertices, 3)
            # normals_view: (batch_size, num_vertices, 3)
            pos_clip, pos_view, normals_view = self.transform_pos(v_posed, n_posed, intrinsic,
                                                                  extrinsic, inputs['img_res'])
            normals_view = nn.functional.normalize(normals_view, dim=-1)

            # render vertex attributes and texture to frame buffer
            # attr_buffer: (batch_size, H, W, 6 + 3*50 + 36*3 + num_bones)
            # tex_buffer: (batch_size, H, W, dim_pos_feature+num_texture_basis*(dim_radiance_feature+1))
            rad_tex = self.radiance_textures[i]
            if self.force_inner_alpha_full and i == 0:
                rad_tex = torch.cat([rad_tex[..., :-1], torch.ones_like(rad_tex[..., -1:])], dim=-1)
            elif self.same_alpha_across_basis:
                rad_tex_alpha = self.radiance_textures_alpha[i]
                rad_tex = torch.cat(
                    [rad_tex[..., :-1],
                     rad_tex_alpha.expand(rad_tex.shape[0], -1, -1, -1)], dim=-1)
            rad_tex = rad_tex.permute(1, 2, 0, 3).flatten(2, 3)
            attr_buffer, tex_buffer, mask = self.render(
                pos=pos_clip,
                attrs=torch.cat([pos_view, normals_view] + ([
                    shapedirs.flatten(2, 3),
                    posedirs.flatten(2, 3),
                    lbs_weights,
                ] if self.rasterize_flame_weights else []), -1),
                uvs=self.uvs[i],
                tri=self.faces[i],
                tri_uv=self.faces_uv[i],
                tex=torch.cat([
                    fake_quant(self.position_texture[i], scale=255, signed=False),
                    fake_quant(rad_tex, scale=255, signed=False),
                ] if self.enable_texture_qat else [self.position_texture[i], rad_tex], -1),
                img_res=inputs['img_res'],
            )

            # dist: (batch_size, H, W)
            # viewdir: (batch_size, H, W, 3)
            # normal: (batch_size, H, W, 3)
            # shapedirs: (batch_size, H, W, 3, dim_expression)
            # posedirs: (batch_size, H, W, 36, 3)
            # lbs_weights: (batch_size, H, W, num_bones)
            # pos_feat: (batch_size, H, W, dim_pos_feature)
            # rad_feat: (batch_size, H, W, num_texture_basis*(dim_radiance_feature+1))
            dist = torch.norm(attr_buffer[..., :attr_dims[0]], dim=-1)
            viewdir = nn.functional.normalize(attr_buffer[..., :attr_dims[0]], dim=-1)
            normal = nn.functional.normalize(attr_buffer[..., attr_dims[0]:attr_dims[1]], dim=-1)
            if self.rasterize_flame_weights:
                shapedirs = attr_buffer[..., attr_dims[1]:attr_dims[2]].reshape(*mask.shape, 3, -1)
                posedirs = attr_buffer[..., attr_dims[2]:attr_dims[3]].reshape(*mask.shape, 36, 3)
                lbs_weights = attr_buffer[..., attr_dims[3]:].reshape(*mask.shape, -1)
            pos_feat = tex_buffer[..., :self.dim_pos_feature]
            rad_feat = tex_buffer[..., self.dim_pos_feature:].reshape(*mask.shape,
                                                                      self.num_texture_basis,
                                                                      self.dim_radiance_feature + 1)
            feature = rad_feat[..., :-1]  # (batch_size, H, W, num_texture_basis, 3)
            alpha = rad_feat[..., -1]  # (batch_size, H, W, num_texture_basis)

            # project spatially varing feature and global conditions to texture coefficients
            # texture_coef: (batch_size, H, W, num_texture_basis)
            texture_coef = torch.zeros(*mask.shape, self.num_texture_basis, device=device)
            texture_coef[mask] = self.tex_projector(
                pos_feat[mask],
                normal[mask],
                viewdir[mask],
                pose[:, None, None, 6:15].expand(*mask.shape, -1)[mask],
                expression[:, None, None, :].expand(*mask.shape, -1)[mask],
                frame_latent[:, None, None, :].expand(*mask.shape, -1)[mask],
            )

            # composite color and alpha
            feature = torch.sum(feature * texture_coef[..., None], dim=3)  # (batch_size, H, W, 3)
            alpha = torch.sum(alpha * texture_coef, dim=3)  # (batch_size, H, W)

            # append results
            features.append(feature.flatten(1, 2))
            alphas.append(alpha.flatten(1, 2))
            dists.append(dist.flatten(1, 2))
            masks.append(mask.flatten(1, 2))
            normals.append(normal.flatten(1, 2))
            if self.rasterize_flame_weights:
                shapedirs_list.append(shapedirs.flatten(1, 2))
                posedirs_list.append(posedirs.flatten(1, 2))
                lbs_weights_list.append(lbs_weights.flatten(1, 2))
            texture_coefs.append(texture_coef.flatten(1, 2))

        # integrate radiance features along using alpha compositing
        feature = torch.stack(features, dim=2)  # (batch_size, num_rays, num_layers, 3)
        alpha = torch.stack(alphas, dim=2)  # (batch_size, num_rays, num_layers)
        dist = torch.stack(dists, dim=2)  # (batch_size, num_rays, num_layers)
        mask = torch.stack(masks, dim=2)  # (batch_size, num_rays, num_layers)
        normal = torch.stack(normals, dim=2)  # (batch_size, num_rays, num_layers, 3)
        if self.rasterize_flame_weights:
            shapedirs = torch.stack(shapedirs_list, dim=2)
            posedirs = torch.stack(posedirs_list, dim=2)
            lbs_weights = torch.stack(lbs_weights_list, dim=2)
        else:
            shapedirs = posedirs = lbs_weights = None
        texture_coef = torch.stack(texture_coefs, dim=2)
        (rgb, ), depth, weight_bg = alpha_integration((feature, ), dist, alpha, mask,
                                                      self.far_distance)
        assert self.dim_radiance_feature == 3, \
            "radiance decoder is not used, but dim_radiance_feature is not 3"

        # blend background color
        rgb = rgb * (1 - weight_bg[:, :, None]) \
            + inputs['background_rgb'][:, None, :] * weight_bg[:, :, None]

        return {
            "rgb": rgb,
            "alpha": (1 - weight_bg),
            "depth": depth,
            "mask": torch.any(mask, dim=-1),
            "mask_manifold": mask,
            "alpha_manifold": alpha,
            "feature_manifold": feature,
            "texture_coef": texture_coef,
            "normal": normal,
            "dist": dist,
            "img_res": inputs['img_res'],
            # FLAME blendshape and lbs weights from mesh vertices
            "shapedirs": shapedirs,
            "posedirs": posedirs,
            "lbs_weights": lbs_weights,
            # GT FLAME blendshape and lbs weights
            "flame_distances": None,
            "flame_index": None,
            "flame_shapedirs": None,
            "flame_posedirs": None,
            "flame_lbs_weights": None,
        }