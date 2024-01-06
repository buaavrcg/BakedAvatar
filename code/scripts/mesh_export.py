import numpy as np
import torch
import tqdm
import time
import math
import os
from accelerate import Accelerator

from flame import FLAME
from utils.training_util import seed_everything
from utils.misc_util import ensure_dir, find_latest_model_path, construct_class_by_name, Logger


def get_flame_template_mesh(flame_model: FLAME.FLAME,
                            num_subdivision=0,
                            with_mouth=True,
                            use_canonical_verts=False):
    """
    Get FLAME template mesh.
    Args:
        flame_model (FLAME): flame model instance
        num_subdivision (int): number of subdivision iterations
        with_mouth (bool): whether to include mouth faces
        use_canonical_verts (bool): use canonical vertices (mouse half opened)
    Returns: mesh data of N vertices and F faces, containing:
        vertices (torch.FloatTensor): (N, 3) vertex positions
        faces (torch.LongTensor): (F, 3) face vertex indices
        shapedirs (torch.FloatTensor): float (N, 50, 3) shape basis
        posedirs (torch.FloatTensor): float (N, 36, 3) pose basis
        lbs_weights (torch.FloatTensor): float (N, 5) shape weights
        normals (torch.FloatTensor): float (N, 3) vertices normals
        uvs (torch.FloatTensor): float (T, 2) vertex uv coordinates
        faces_uv (torch.LongTensor): (F, 3) face uv indices in [0, T)
    """
    from pytorch3d.structures import Meshes
    from pytorch3d.io import load_obj

    N = flame_model.v_template.shape[0]
    if use_canonical_verts:
        vertices = flame_model.canonical_verts.float().squeeze(0)
    else:
        vertices = flame_model.v_template.float()
    # faces = flame_model.faces_tensor.long()
    shapedirs = flame_model.shapedirs[:, :, flame_model.n_shape:].permute(0, 2, 1).float()
    posedirs = flame_model.posedirs.reshape(-1, N, 3).permute(1, 0, 2).float()
    lbs_weights = flame_model.lbs_weights.float()

    # Use faces in template mesh obj file
    template_obj_path = os.path.join(FLAME.FLAME_DIR, "head_template_mesh_mouth.obj" \
                                     if with_mouth else "head_template_mesh.obj")
    with open(template_obj_path, 'r') as f:
        _, faces_data, aux = load_obj(f, load_textures=False)
        uvs = aux.verts_uvs.to(vertices.device).float()
        faces = faces_data.verts_idx.to(vertices.device).long()
        faces_uv = faces_data.textures_idx.to(vertices.device).long()
        assert uvs.shape[0] == torch.max(faces_uv).cpu().item() + 1

    meshes = Meshes(verts=[vertices], faces=[faces])

    # Calculate vertex normals
    normals = torch.zeros_like(vertices)
    face_normals = meshes.faces_normals_packed()  # (F, 3)
    normals[faces[:, 0]] += face_normals
    normals[faces[:, 1]] += face_normals
    normals[faces[:, 2]] += face_normals
    normals = torch.nn.functional.normalize(normals, dim=1)

    # Subdivide the template mesh if needed
    if num_subdivision > 0:
        from pytorch3d.ops import SubdivideMeshes
        mesh_feats = torch.cat([
            normals,
            shapedirs.reshape(N, -1),
            posedirs.reshape(N, -1),
            lbs_weights.reshape(N, -1),
        ], 1)
        subdiv = SubdivideMeshes()
        uvmeshes = Meshes(verts=[uvs], faces=[faces_uv])
        for _ in range(num_subdivision):
            meshes, mesh_feats = subdiv(meshes, mesh_feats)
            uvmeshes = subdiv(uvmeshes)
        vertices, faces = meshes.verts_packed(), meshes.faces_packed()
        uvs, faces_uv = uvmeshes.verts_packed(), uvmeshes.faces_packed()
        idx = np.cumsum([3, shapedirs[0].numel(), posedirs[0].numel()])
        normals = mesh_feats[:, :idx[0]].reshape(-1, *normals.shape[1:])
        normals = torch.nn.functional.normalize(normals, dim=1)
        shapedirs = mesh_feats[:, idx[0]:idx[1]].reshape(-1, *shapedirs.shape[1:])
        posedirs = mesh_feats[:, idx[1]:idx[2]].reshape(-1, *posedirs.shape[1:])
        lbs_weights = mesh_feats[:, idx[2]:].reshape(-1, *lbs_weights.shape[1:])

    return vertices, faces, shapedirs, posedirs, lbs_weights, normals, uvs, faces_uv


def save_mesh_data(output_dir,
                   levels,
                   vertices,
                   faces,
                   shapedirs,
                   posedirs,
                   lbs_weights,
                   normals,
                   uvs,
                   faces_uv,
                   flame_model: FLAME.FLAME,
                   texture_maps=None):
    """Save mesh data to output directory as a .pkl file and preview obj/ply files."""
    from pathlib import Path
    from pytorch3d.io import save_ply, save_obj

    ensure_dir(output_dir, False)
    ensure_dir(os.path.join(output_dir, "ply"), False)
    ensure_dir(os.path.join(output_dir, "obj"), False)

    meshes_data = []
    for i, (l, V, F, E, P, W, N, UV, FUV) in tqdm.tqdm(enumerate(
            zip(levels, vertices, faces, shapedirs, posedirs, lbs_weights, normals, uvs, faces_uv)),
                                                       desc="Saving meshes"):

        if texture_maps is not None:
            TEX = texture_maps[i]
            assert TEX.shape[-1] == 3, "Texture map must be RGB"
        else:
            TEX = torch.ones(64, 64, 3, device=UV.device)

        save_ply(Path(os.path.join(output_dir, "ply", f"mesh_{i}_canonical.ply")), V, F, N)
        save_obj(Path(os.path.join(output_dir, "obj", f"mesh_{i}_canonical.obj")),
                 verts=V,
                 faces=F,
                 verts_uvs=UV,
                 faces_uvs=FUV,
                 texture_map=TEX)

        # convert to original FLAME pose
        V, N = flame_model.get_original_points(V, N, E.transpose(1, 2), P, W)

        meshes_data.append({
            'level': l.cpu().item(),
            'vertices': V.cpu().numpy().astype(np.float32),
            'faces': F.cpu().numpy().astype(np.int32),
            'shapedirs': E.cpu().numpy().astype(np.float32),
            'posedirs': P.cpu().numpy().astype(np.float32),
            'lbs_weights': W.cpu().numpy().astype(np.float32),
            'normals': N.cpu().numpy().astype(np.float32),
            'uvs': UV.cpu().numpy().astype(np.float32),
            'faces_uv': FUV.cpu().numpy().astype(np.int32),
        })

        save_ply(Path(os.path.join(output_dir, "ply", f"mesh_{i}_original.ply")), V, F, N)
        save_obj(Path(os.path.join(output_dir, "obj", f"mesh_{i}_original.obj")),
                 verts=V,
                 faces=F,
                 verts_uvs=UV,
                 faces_uvs=FUV,
                 texture_map=TEX)

    common_data = {
        'v_template': flame_model.v_template.cpu().numpy().astype(np.float32),
        'shapedirs': flame_model.shapedirs[..., flame_model.n_shape:] \
            .cpu().numpy().astype(np.float32),
        'posedirs': flame_model.posedirs.cpu().numpy().astype(np.float32),
        'lbs_weights': flame_model.lbs_weights.cpu().numpy().astype(np.float32),
        'J_regressor': flame_model.J_regressor.cpu().numpy().astype(np.float32),
        'parents': flame_model.parents.cpu().numpy().astype(np.int32),
    }
    torch.save({
        'common': common_data,
        'meshes': meshes_data,
        'ghostbone': W.shape[-1] > 5,
    }, os.path.join(output_dir, "mesh_data.pkl"))


def clean_mesh(vertices: np.ndarray, faces: np.ndarray):
    """Clean mesh by removing non-manifold vertices and faces."""
    from trimesh import Trimesh
    trimesh = Trimesh(vertices, faces)
    trimesh = max(trimesh.split(only_watertight=False), key=lambda c: c.area)
    vertices, faces = trimesh.vertices, trimesh.faces
    return vertices, faces


def decimate_mesh(vertices: np.ndarray,
                  faces: np.ndarray,
                  target_face_count,
                  backend="pymeshlab",
                  preserve_border=False,
                  optimalplacement=True) -> tuple[np.ndarray, np.ndarray]:
    """Decimate excess mesh faces to target face count."""
    if backend == "pyfqmr":
        import pyfqmr
        mesh_simplifer = pyfqmr.Simplify()
        mesh_simplifer.setMesh(vertices, faces)
        mesh_simplifer.simplify_mesh(target_count=target_face_count,
                                     preserve_border=preserve_border,
                                     verbose=10)
        vertices, faces, _ = mesh_simplifer.getMesh()
    elif backend == "pymeshlab":
        import pymeshlab
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertices, faces), 'mesh')
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=target_face_count,
                                                    preserveboundary=preserve_border,
                                                    optimalplacement=optimalplacement)
        mesh = ms.current_mesh()
        vertices, faces = mesh.vertex_matrix(), mesh.face_matrix()
    else:
        assert 0, f"Unknown decimate_mesh() backend: {backend}"

    return vertices, faces


def filter_mesh_by_normal(vertices: np.ndarray,
                          faces: np.ndarray,
                          mesh_angle_limit: float,
                          viewpoint=[0, 0, -10],
                          expand_iterations=3,
                          delete_faces=False,
                          target_face_percentage=0.5,
                          remesh_iteration=0,
                          remesh_target_len_percentage=10.0):
    """Filter mesh faces by angle between their normal and the view direction."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces), 'mesh')
    ms.compute_selection_by_angle_with_direction_per_face(anglelimit=mesh_angle_limit,
                                                          viewpoint=np.array(viewpoint))

    for _ in range(expand_iterations):
        ms.compute_selection_transfer_face_to_vertex(inclusive=False)
        ms.compute_selection_by_condition_per_face(condselect='fsel || vsel0 || vsel1 || vsel2')

    if delete_faces:
        ms.meshing_remove_selected_faces()
        ms.meshing_remove_unreferenced_vertices()
    else:
        ms.meshing_decimation_quadric_edge_collapse(targetperc=target_face_percentage,
                                                    qualitythr=0.5,
                                                    optimalplacement=True,
                                                    selected=True)
        if remesh_iteration > 0:
            ms.meshing_isotropic_explicit_remeshing(
                iterations=remesh_iteration,
                targetlen=pymeshlab.PercentageValue(remesh_target_len_percentage),
                selectedonly=True)

    mesh = ms.current_mesh()
    vertices, faces = mesh.vertex_matrix(), mesh.face_matrix()

    vertices, faces = clean_mesh(vertices, faces)
    return vertices, faces


def optimize_mesh(
    vertices: np.ndarray,
    faces: np.ndarray,
    vertices_merge_percentage=10.0,
    min_diameter_percentage=10.0,
    min_face_number=100,
    remesh_iteration=3,
    remesh_target_len_percentage=1.0,
):
    """Optimize mesh by removing unreferenced vertices, duplicate faces, null faces, and
       small components. Optionally remesh the mesh to remove degenerate triangles."""
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertices, faces), 'mesh')

    # apply pymeshlab filters
    ms.meshing_remove_unreferenced_vertices()

    if vertices_merge_percentage > 0:
        ms.meshing_merge_close_vertices(threshold=pymeshlab.PercentageValue(vertices_merge_percentage))

    ms.meshing_remove_duplicate_faces()  # faces defined by the same verts
    ms.meshing_remove_null_faces()  # faces with area == 0

    if min_diameter_percentage > 0:
        ms.meshing_remove_connected_component_by_diameter(
            mincomponentdiag=pymeshlab.PercentageValue(min_diameter_percentage))

    if min_face_number > 0:
        ms.meshing_remove_connected_component_by_face_number(mincomponentsize=min_face_number)

    if remesh_iteration > 0:
        ms.meshing_isotropic_explicit_remeshing(
            iterations=remesh_iteration,
            adaptive=True,
            targetlen=pymeshlab.PercentageValue(remesh_target_len_percentage))

    mesh = ms.current_mesh()
    vertices, faces = mesh.vertex_matrix(), mesh.face_matrix()

    return vertices, faces


def apply_mesh_simplification(vertices: np.ndarray,
                              faces: np.ndarray,
                              mesh_angle_limit=None,
                              mesh_delete_high_angle=False,
                              mesh_high_angle_face_percentage=0.1,
                              mesh_high_angle_expand_iterations=3,
                              mesh_target_face_count=None,
                              mesh_remesh_iteration=0,
                              mesh_remesh_target_len_percentage=1.0):
    print(f"Initial mesh vertices: {len(vertices)}, faces: {len(faces)}")

    # clean mesh
    vertices, faces = clean_mesh(vertices, faces)
    print(f"Cleaned vertices: {len(vertices)}, faces: {len(faces)}")

    # decimate mesh faces to target count
    if mesh_target_face_count is not None:
        vertices, faces = decimate_mesh(vertices, faces, mesh_target_face_count)
        print(f"Decimated vertices: {len(vertices)}, faces: {len(faces)}")

    # only keep faces that face front to z+ axis
    if mesh_angle_limit is not None:
        vertices, faces = filter_mesh_by_normal(
            vertices,
            faces,
            mesh_angle_limit,
            expand_iterations=mesh_high_angle_expand_iterations,
            delete_faces=mesh_delete_high_angle,
            target_face_percentage=mesh_high_angle_face_percentage,
        )
        print(f"Normal filtered vertices: {len(vertices)}, faces: {len(faces)}")

    # optimize mesh and remeshing
    vertices, faces = optimize_mesh(
        vertices,
        faces,
        remesh_iteration=mesh_remesh_iteration,
        remesh_target_len_percentage=mesh_remesh_target_len_percentage,
    )
    print(f"Optimized vertices: {len(vertices)}, faces: {len(faces)}")

    return vertices, faces


def apply_uv_parameterization(vertices: np.ndarray, faces: np.ndarray):
    import xatlas

    # parameterize mesh and expand UV using xatlas
    atlas = xatlas.Atlas()
    atlas.add_mesh(vertices, faces)

    chart_options = xatlas.ChartOptions()
    chart_options.max_iterations = 50
    chart_options.max_cost = 1000.0

    pack_options = xatlas.PackOptions()
    pack_options.create_image = True
    pack_options.padding = 1
    pack_options.bilinear = True
    pack_options.bruteForce = True
    pack_options.rotate_charts = True

    atlas.generate(chart_options=chart_options, pack_options=pack_options)
    vmapping, faces_uv, uvs = atlas[0]
    faces_uv = faces_uv.astype(np.int32)
    texture_map = np.flip(atlas.chart_image, axis=0) / 255
    print(f"Atlas generated: {atlas.width}x{atlas.height}, "
          f"utilization: {atlas.utilization}, chart count: {atlas.chart_count}, "
          f"number of uv coords: {len(uvs)}")

    return uvs, faces_uv, texture_map


def export_flame_template_mesh(output_dir, model, num_subdivision):
    vertices, faces, shapedirs, posedirs, lbs_weights, normals, uvs, faces_uv \
        = get_flame_template_mesh(model.flame, num_subdivision, use_canonical_verts=True)

    save_mesh_data(
        os.path.join(output_dir, "flame_template"),
        torch.tensor([0.0], device=vertices.device),
        vertices.unsqueeze(0),
        faces.unsqueeze(0),
        shapedirs.unsqueeze(0),
        posedirs.unsqueeze(0),
        lbs_weights.unsqueeze(0),
        normals.unsqueeze(0),
        uvs.unsqueeze(0),
        faces_uv.unsqueeze(0),
        model.flame,
    )


def flame_fitting(
    accel,
    output_dir,
    model,
    dataset,
    iterations,
    lr,
    num_subdivision,
    num_sample_points,
    loss_class,
    loss_args,
    optim_class,
    optim_args,
    scheduler_class,
    scheduler_args,
    show_it,
    save_it,
):
    from pytorch3d.structures import Meshes
    from pytorch3d.ops import sample_points_from_meshes

    vertices, faces, _, _, _, _, uvs, faces_uv \
        = get_flame_template_mesh(model.flame, num_subdivision, True, True)

    num_points_per_view = math.ceil(num_sample_points / len(dataset))
    points = sample_surface_pointcloud(accel, model, dataset, num_points_per_view)

    # optimize mesh vertices
    target_levels = model.renderer.manifold_levels.data.to(accel.device)
    num_levels = target_levels.shape[0]
    num_vertices = vertices.shape[0]
    src_meshes = Meshes(
        vertices.repeat(num_levels, 1, 1),
        faces.repeat(num_levels, 1, 1),
    ).to(accel.device)
    verts_offsets = torch.nn.Parameter(torch.zeros_like(src_meshes.verts_packed()), True)
    accel.print(f"Parameters total: {verts_offsets.numel()}")

    loss = construct_class_by_name(class_name=loss_class, **loss_args)
    optimizer = construct_class_by_name([verts_offsets],
                                        class_name=optim_class,
                                        lr=lr,
                                        **optim_args)
    scheduler = construct_class_by_name(optimizer, class_name=scheduler_class, **scheduler_args)
    loss, optimizer, scheduler = accel.prepare(loss, optimizer, scheduler)
    accel.print(f"Start FLAME fitting for {iterations} iterations...")

    edges = Meshes(verts=[vertices], faces=[faces]).to(accel.device).edges_packed()
    targets = {
        'level': target_levels,
        'initial_vertices': vertices,
        'faces': faces,
        'edges': edges,
        'edges_length': torch.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], dim=1),
        'points': [level_points.to(accel.device) for level_points in points],
    }
    num_samples = 5000

    last_it, last_time = 0, time.time()
    for it in range(iterations):
        optimizer.zero_grad()

        new_meshes = src_meshes.offset_verts(verts_offsets)
        verts_points = new_meshes.verts_padded()
        verts_normals = new_meshes.verts_normals_padded()

        if num_samples > 0:
            sampled_points, sampled_normals = \
                sample_points_from_meshes(new_meshes, num_samples, True)
            points = torch.cat([verts_points, sampled_points], dim=1)
            normals = torch.cat([verts_normals, sampled_normals], dim=1)
        else:
            points, normals = verts_points, verts_normals
            sampled_points, sampled_normals = None, None

        scalar, gradient = model.query_canonical_manifold(points.view(-1, 3))
        scalar = scalar.view(num_levels, num_vertices + num_samples)
        gradient = gradient.view(num_levels, num_vertices + num_samples, 3)

        verts_scalar, verts_gradient = scalar[:, :num_vertices], gradient[:, :num_vertices]
        sampled_scalar, sampled_gradient = scalar[:, num_vertices:], gradient[:, num_vertices:]

        outputs = {
            'meshes': new_meshes,
            'verts_points': verts_points,  # (num_levels, N, 3)
            'verts_normals': verts_normals,  # (num_levels, N, 3)
            'verts_scalar': verts_scalar,  # (num_levels, N)
            'verts_gradient': verts_gradient,  # (num_levels, N, 3)
            'sampled_points': sampled_points,  # (num_levels, num_samples, 3)
            'sampled_normals': sampled_normals,  # (num_levels, num_samples, 3)
            'sampled_scalar': sampled_scalar,  # (num_levels, num_samples)
            'sampled_gradient': sampled_gradient,  # (num_levels, num_samples, 3)
            'points': points,  # (num_levels, N + num_samples, 3)
            'normals': normals,  # (num_levels, N + num_samples, 3)
            'scalar': scalar,  # (num_levels, N + num_samples)
            'gradient': gradient,  # (num_levels, N + num_samples, 3)
        }
        loss_total, loss_dict = loss(outputs, targets)
        accel.backward(loss_total)
        optimizer.step()
        scheduler.step()

        if it % show_it == 0 and accel.is_local_main_process:
            elasped = time.time() - last_time
            num_it = it - last_it
            speed = num_it / elasped
            loss_total = loss_dict.pop('loss')
            accel.print("".join([
                f"[{it:07d}][{elasped:.2f}s][{speed:.2f}it/s]",
                f" total: {loss_total:.4f}",
                *list(f", {n[5:] if n[:4] == 'loss' else n}: {v:.4f}"
                      for n, v in sorted(loss_dict.items())),
            ]))
            last_it, last_time = it, time.time()

        if (it % save_it == 0 or it + 1 == iterations) and accel.is_local_main_process:
            # evaluate deformer network at all vertices
            with torch.no_grad():
                E, P, W = model.deformer_net.query_weights(verts_points.view(-1, 3))
                E = E.reshape(num_levels, num_vertices, 3, model.dim_expression).permute(0, 1, 3, 2)
                P = P.reshape(num_levels, num_vertices, 36, 3)
                W = W.reshape(num_levels, num_vertices, model.deformer_net.num_bones)
                _, gradients = model.query_canonical_manifold(verts_points.detach().view(-1, 3))
                gradients = gradients.view(num_levels, num_vertices, 3)
                normals = torch.nn.functional.normalize(gradients, dim=-1)

            # save meshes data
            save_dir = os.path.join(output_dir, 'flame_fitting', f'iter_{it}')
            ensure_dir(save_dir, False)
            save_mesh_data(
                save_dir,
                target_levels,
                verts_points.detach(),
                faces.repeat(num_levels, 1, 1),
                E,
                P,
                W,
                normals.detach(),
                uvs.repeat(num_levels, 1, 1),
                faces_uv.repeat(num_levels, 1, 1),
                model.flame,
            )


def marching_cube(
    output_dir,
    model,
    device,
    # Marching cube settings
    mc_res_init,
    mc_res_up,
    mc_query_scale,
    mc_center_offset,
    # Mesh simplification settings
    **mesh_kwargs,
):
    import mise
    from skimage.measure import marching_cubes
    print(f"Marching cube final resolution: {mc_res_init * (2**mc_res_up)}")

    template_vertices = get_flame_template_mesh(model.flame, 0, False, True)[0]
    template_bbox = torch.stack(
        [torch.min(template_vertices, dim=0)[0],
         torch.max(template_vertices, dim=0)[0]])
    template_center = (template_bbox[0] + template_bbox[1]) / 2
    template_center += torch.tensor(mc_center_offset, device=device, dtype=torch.float32)
    template_size = torch.max(template_bbox[1] - template_bbox[0])

    vertices_list = []
    faces_list = []
    shapedirs_list = []
    posedirs_list = []
    lbs_weights_list = []
    normals_list = []
    uvs_list = []
    faces_uv_list = []
    texture_map_list = []
    target_levels = model.renderer.manifold_levels.data.to(device)
    for i in range(target_levels.shape[0]):
        level = target_levels[i].cpu().item()
        print(f"Marching cube mesh {i}: level={level}")

        # use MISE to query the manifold, and get the dense levelset grid
        mesh_extractor = mise.MISE(mc_res_init, mc_res_up, level)
        query_points = mesh_extractor.query()
        while query_points.shape[0] > 0:
            points = torch.from_numpy(query_points.astype(np.float32)).to(device)
            points = points / mesh_extractor.resolution - 0.5  # normalize to [-0.5, 0.5]
            points = points * (template_size * mc_query_scale) + template_center
            with torch.no_grad():
                scalar = model.query_canonical_manifold(points, no_gradient=True).squeeze(1)
            mesh_extractor.update(query_points, scalar.cpu().numpy().astype(np.float64))
            query_points = mesh_extractor.query()

        # run marching cube
        levelset_volume = mesh_extractor.to_dense().astype(np.float32)
        vertices, faces, _, _ = marching_cubes(levelset_volume,
                                               level,
                                               gradient_direction='descent',
                                               allow_degenerate=False)
        vertices = vertices / mesh_extractor.resolution - 0.5  # in [-0.5, 0.5]
        vertices = vertices * (template_size.cpu().numpy() * mc_query_scale) \
                 + template_center.cpu().numpy()

        # simplify mesh and expand uv
        vertices, faces = apply_mesh_simplification(vertices, faces, **mesh_kwargs)
        uvs, faces_uv, texture_map = apply_uv_parameterization(vertices, faces)

        # evaluate deformer network at all vertices
        vertices = torch.from_numpy(vertices).float().to(device)
        faces = torch.from_numpy(faces).long().to(device)
        uvs = torch.from_numpy(uvs).float().to(device)
        faces_uv = torch.from_numpy(faces_uv).long().to(device)
        texture_map = torch.from_numpy(texture_map).float().to(device)
        num_vertices = vertices.shape[0]
        with torch.no_grad():
            E, P, W = model.deformer_net.query_weights(vertices.view(-1, 3))
            E = E.reshape(num_vertices, 3, model.dim_expression).permute(0, 2, 1)
            P = P.reshape(num_vertices, 36, 3)
            W = W.reshape(num_vertices, model.deformer_net.num_bones)
            _, gradients = model.query_canonical_manifold(vertices.view(-1, 3))
            gradients = gradients.detach().view(num_vertices, 3)
            normals = torch.nn.functional.normalize(gradients, dim=-1)

        # save mesh
        vertices_list.append(vertices.cpu())
        faces_list.append(faces.cpu())
        shapedirs_list.append(E.cpu())
        posedirs_list.append(P.cpu())
        lbs_weights_list.append(W.cpu())
        normals_list.append(normals.cpu())
        uvs_list.append(uvs.cpu())
        faces_uv_list.append(faces_uv.cpu())
        texture_map_list.append(texture_map.cpu())

    # save meshes data
    save_dir = os.path.join(output_dir, 'marching_cube', f'res_init{mc_res_init}_up{mc_res_up}')
    ensure_dir(save_dir, False)
    save_mesh_data(
        save_dir,
        target_levels,
        vertices_list,
        faces_list,
        shapedirs_list,
        posedirs_list,
        lbs_weights_list,
        normals_list,
        uvs_list,
        faces_uv_list,
        model.flame.cpu(),
        texture_maps=texture_map_list,
    )


def mesh_export(
    rundir,
    seed,
    use_cpu,
    # Dataset (only for loading shape params)
    dataset_class,
    data_dir,
    train_subdirs,
    img_res,
    train_subsample,
    # Model
    model_class,
    model_args,
    # Exporting
    export_iteration,
    export_type,
    # FLAME mesh settings,
    flame_num_subdivision,
    flame_sample_points,
    flame_fitting_loss_class,
    flame_fitting_loss_args,
    # Marching cube settings
    mc_res_init,
    mc_res_up,
    mc_query_scale,
    mc_center_offset_x,
    mc_center_offset_y,
    mc_center_offset_z,
    # Optimizer & Scheduler
    optim_class,
    optim_args,
    scheduler_class,
    scheduler_args,
    # Fitting hyperparameters
    fitting_iterations,
    fitting_lr,
    # Logging
    show_it,
    save_it,
    **kwargs,
):
    accel = Accelerator(cpu=use_cpu)
    seed_everything(seed + accel.process_index)  # set seed
    checkpoints_dir = os.path.join(rundir, "checkpoints")
    if accel.is_local_main_process:
        assert os.path.exists(checkpoints_dir), "No checkpoints found!"
        Logger(os.path.join(rundir, "exporting_log.txt"), "w+")

    # Load dataset
    train_dataset = construct_class_by_name(class_name=dataset_class,
                                            data_dir=data_dir,
                                            sub_dirs=train_subdirs,
                                            img_res=img_res,
                                            num_rays=-1,
                                            subsample=train_subsample,
                                            use_semantics=False,
                                            no_gt=False)
    accel.print(f"Loaded {len(train_dataset)} testing frames from {data_dir}/{train_subdirs}.")

    # Build and load model
    model = construct_class_by_name(class_name=model_class,
                                    shape_params=train_dataset.get_shape_params(),
                                    canonical_exp=train_dataset.get_mean_expression(),
                                    **model_args)
    if export_iteration is not None:
        load_ckpt_dir = os.path.join(checkpoints_dir, f"iter_{export_iteration}")
    else:
        load_ckpt_dir = find_latest_model_path(checkpoints_dir)
    model_state = torch.load(os.path.join(load_ckpt_dir, "model.pth"), accel.device)
    model.load_state_dict(model_state['model'], strict=True)
    it, sample_count = model_state['it'], model_state['sample_count']
    accel.print(f'Loaded checkpoint (iter {it}, samples {sample_count}) from: {load_ckpt_dir}')
    model = accel.prepare(model)

    # start exporting mesh
    accel.print(f"Start exporting mesh (type: {export_type})...")
    model.eval()
    output_dir = os.path.join(rundir, "mesh_export", f"iter_{it}")
    ensure_dir(output_dir, False)

    mesh_kwargs = {k: v for k, v in kwargs.items() if k.startswith('mesh_')}

    if export_type == "flame_template":
        export_flame_template_mesh(output_dir, model, flame_num_subdivision)
    elif export_type == "flame_fitting":
        flame_fitting(
            accel,
            output_dir,
            model,
            train_dataset,
            fitting_iterations,
            fitting_lr,
            flame_num_subdivision,
            flame_sample_points,
            flame_fitting_loss_class,
            flame_fitting_loss_args,
            optim_class,
            optim_args,
            scheduler_class,
            scheduler_args,
            show_it,
            save_it,
        )
    elif export_type == "marching_cube":
        marching_cube(
            output_dir,
            model,
            accel.device,
            mc_res_init,
            mc_res_up,
            mc_query_scale,
            [mc_center_offset_x, mc_center_offset_y, mc_center_offset_z],
            **mesh_kwargs,
        )
    else:
        assert 0, f"Unknown export type: {export_type}"
