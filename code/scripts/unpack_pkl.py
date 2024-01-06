import numpy as np
import torch
import os
import sys
import json
import imageio
import argparse

root_path = os.path.realpath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root_path)

from utils.misc_util import ensure_dir

ATLAS_SIZE = 4096
SAVE_PNG = False


def write_ndarray(v: np.ndarray, name: str, output_dir: str, root_dir: str):
    if SAVE_PNG and v.ndim == 3 and v.dtype == np.uint8 and (v.shape[-1] == 3 or v.shape[-1] == 4):
        path = os.path.join(output_dir, f"{name}.png")
        imageio.imsave(path, v)
    else:
        path = os.path.join(output_dir, f"{name}.npy")
        np.save(path, np.ascontiguousarray(v))

    return {
        "path": os.path.relpath(path, root_dir),
        "shape": v.shape,
        "dtype": str(v.dtype),
    }


def duplicate_uv_faces(mesh_dict) -> dict:
    V = mesh_dict.pop("vertices")
    F = mesh_dict.pop("faces")
    E = mesh_dict.pop("shapedirs")
    P = mesh_dict.pop("posedirs")
    W = mesh_dict.pop("lbs_weights")
    N = mesh_dict.pop("normals")
    UV = mesh_dict.pop("uvs")
    FUV = mesh_dict.pop("faces_uv")

    # use the same faces indices for both vertices and uvs
    assert F.shape == FUV.shape
    if np.any(F != FUV):
        NV = np.zeros((UV.shape[0], *V.shape[1:]), dtype=V.dtype)
        NE = np.zeros((UV.shape[0], *E.shape[1:]), dtype=E.dtype)
        NP = np.zeros((UV.shape[0], *P.shape[1:]), dtype=P.dtype)
        NW = np.zeros((UV.shape[0], *W.shape[1:]), dtype=W.dtype)
        NN = np.zeros((UV.shape[0], *N.shape[1:]), dtype=N.dtype)
        NV[FUV] = V[F]
        NE[FUV] = E[F]
        NP[FUV] = P[F]
        NW[FUV] = W[F]
        NN[FUV] = N[F]
        V, E, P, W, N = NV, NE, NP, NW, NN
        F = FUV

    # make position texture atlas
    position_texture = mesh_dict.pop("position_texture")
    height, width, channels = position_texture.shape
    num_textures = (channels + 3) // 4
    assert ATLAS_SIZE % height == 0 and ATLAS_SIZE % width == 0
    textures_per_row = ATLAS_SIZE // width
    textures_per_column = ATLAS_SIZE // height
    assert num_textures <= textures_per_row * textures_per_column, \
        f"Number of radiance textures exceeds the atlas size {ATLAS_SIZE}"
    num_columns = min(num_textures, textures_per_row)
    num_rows = max(num_textures // textures_per_row, 1)
    position_atlas = np.zeros((height * num_rows, width * num_columns, 4),
                              dtype=position_texture.dtype)
    texture_index = 0
    for yi in range(num_rows):
        for xi in range(num_columns):
            if texture_index >= num_textures:
                break
            x = xi * width
            y = yi * height
            c = texture_index * 4
            position_chunk = position_texture[:, :, c:c + 4]
            if position_chunk.shape[-1] < 4:
                position_chunk = np.pad(position_chunk, [(0, 0), (0, 0),
                                                         (0, 4 - position_chunk.shape[-1])],
                                        mode='constant',
                                        constant_values=0)
            position_atlas[y:y + height, x:x + width] = position_chunk
            texture_index += 1

    # make radiance texture atlas
    radiance_textures = mesh_dict.pop("radiance_textures")
    num_textures, height, width, channels = radiance_textures.shape
    assert ATLAS_SIZE % height == 0 and ATLAS_SIZE % width == 0
    textures_per_row = ATLAS_SIZE // width
    textures_per_column = ATLAS_SIZE // height
    assert num_textures <= textures_per_row * textures_per_column, \
        f"Number of radiance textures exceeds the atlas size {ATLAS_SIZE}"
    num_columns = min(num_textures, textures_per_row)
    num_rows = max(num_textures // textures_per_row, 1)
    radiance_atlas = np.zeros((height * num_rows, width * num_columns, channels),
                              dtype=radiance_textures.dtype)
    texture_index = 0
    for yi in range(num_rows):
        for xi in range(num_columns):
            if texture_index >= num_textures:
                break
            x = xi * width
            y = yi * height
            radiance_atlas[y:y + height, x:x + width] = radiance_textures[texture_index]
            texture_index += 1

    return {
        "vertices": V,
        "faces": F,
        "shapedirs": E,
        "posedirs": P,
        "lbs_weights": W,
        "normals": N,
        "uvs": UV,
        "position_texture": position_atlas,
        "radiance_texture": radiance_atlas,
        **mesh_dict,
    }


def write_data_dict(output_dir, data_dict, root_dir=None) -> dict:
    """
    Write data dict containing multiple numpy array to the output dir,
    and returns a new meta dict with numpy array replaced by their paths.
    """
    ensure_dir(output_dir, False)
    root_dir = output_dir if root_dir is None else root_dir

    for k, v in data_dict.items():
        if isinstance(v, np.ndarray):
            data_dict[k] = write_ndarray(v, f"{k}", output_dir, root_dir)
        elif isinstance(v, (int, float, str)):
            pass
        elif isinstance(v, dict):
            data_dict[k] = write_data_dict(os.path.join(output_dir, k), v, root_dir)
        elif isinstance(v, list):
            for i, vi in enumerate(v):
                if isinstance(vi, np.ndarray):
                    v[i] = write_ndarray(vi, f"{k}_{i}", output_dir, root_dir)
                elif isinstance(vi, dict):
                    if k == "meshes":
                        vi = duplicate_uv_faces(vi)
                    v[i] = write_data_dict(os.path.join(output_dir, f"{k}_{i}"), vi, root_dir)
                else:
                    raise ValueError(f"Cannot write unknown data type in list: {type(vi)}")
        else:
            raise ValueError(f"Cannot write unknown data type: {type(v)}")

    return data_dict


def unpack_pkl_to(pkl_path, output_dir):
    data_dict = torch.load(pkl_path)
    metadata = write_data_dict(output_dir, data_dict)
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Data dict written to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help="Path to the input data dict")
    parser.add_argument('--output', required=False, help="Output directory path")
    parser.add_argument('--save_png', action='store_true', help="Save png for RGBA images")
    args = parser.parse_args()

    assert os.path.exists(args.input), f"Input file {args.input} does not exist"
    if args.output is None:
        args.output = os.path.splitext(args.input)[0]

    SAVE_PNG = bool(args.save_png)
    unpack_pkl_to(args.input, args.output)
