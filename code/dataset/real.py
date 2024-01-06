import torch
import numpy as np
import json
import os
from torch.utils.data import Dataset
from utils.image_util import load_rgb_image, load_mask, load_semantic
from utils.render_util import load_K_Rt_from_P


class RealVideoDataset(Dataset):
    """
    Dataset contains multiple clips of a single persion, each clip is in one of the subdirectories.
    Data structure:
        + RGB images in data_dir/sub_dirs[i]/image
        + Foreground masks in data_dir/sub_dirs[i]/mask
        + (optional) Semantic masks in data_dir/sub_dirs[i]/semantic
        + Json files containing FLAME parameters in data_dir/sub_dirs[i]/flame_params.json
    Json file structure: {
        "frames": [
            {
                "file_path": <relative path to image>,
                "world_mat": <camera extrinsic matrix (world to camera)>, 
                    // Camera rotation is actually the same for all frames, since the camera is fixed during capture.
                    // The FLAME head is centered at the origin, scaled by 4 times.
                "expression": <50 dimension expression parameters>, 
                "pose": <15 dimension pose parameters>,
                "flame_keypoints": <2D facial keypoints calculated from FLAME>
            }
        ],
        "shape_params": <100 dimension FLAME shape parameters, shared by all scripts and testing frames of the subject>,
        "intrinsics": <camera focal length fx, fy and the offsets of the principal point cx, cy>
    }
    """
    def __init__(
        self,
        data_dir,
        sub_dirs,
        img_res,
        num_rays=-1,
        subsample=1,
        use_semantics=False,
        semantic_sampling_ratio=0,
        no_gt=False,
        background_rgb=[1, 1, 1],
        mask_threshold=None,
    ):
        """
        Args:
            data_dir: Path to the directory containing the preprocessed video data.
            sub_dirs: List of subdirectories for the subject, e.g. [MVI_1810, MVI_1811]
            img_res: Image resolution, e.g. (256, 256)
            num_rays: Number of rays to sample from each frame. If -1, use all rays.
            subsample: Subsample the number of frames. Can be a number, a list of frame indices,
                or a string of expression that generates a list of frame indices.
            use_semantics: Whether to use semantic masks as model input.
            no_gt: If true, do not load GT images, masks or semantics.
            background_rgb: RGB color of the background. If none, random background will be used.
            mask_threshold: Use custom mask threshold instead of default value (0.5).
        """
        super().__init__()
        self.data_dir = data_dir
        self.img_res = img_res
        self.num_pixels = img_res[0] * img_res[1]
        self.num_rays = num_rays
        self.subsample = eval(subsample) if isinstance(subsample, str) else subsample
        assert isinstance(self.subsample, (int, list))
        self.use_semantics = use_semantics
        self.semantic_sampling_ratio = semantic_sampling_ratio
        self.no_gt = no_gt
        self.background_rgb = None if background_rgb is None \
            else np.array(background_rgb, np.float32)
        self.mask_threshold = mask_threshold

        self.frame_data = {
            "sub_id": [],  # Frame id in the subdirectory
            "sub_dir": [],  # Subdirectory name
            "image_path": [],  # Path to RGB image
            "mask_path": [],  # Path to mask image
            "extrinsic": [],  # Camera extrinsic 4x4 matrix
            "expression": [],  # FLAME parameters for expression
            "pose": [],  # FLAME parameters for pose
            "bbox": [],  # Bounding box of the face in the image (in pixel units)
            "flame_keypoints": [],  # 2D facial keypoints calculated from FLAME
        }
        if self.use_semantics:
            self.frame_data["semantic_path"] = []

        halfsize_bbox = np.array([img_res[0], img_res[1], img_res[0], img_res[1]]) / 2
        for sub_dir in sub_dirs:
            dir = os.path.join(self.data_dir, sub_dir)
            assert os.path.exists(dir), f"Directory {dir} does not exist."

            # Load Camera and FLAME parameters
            with open(os.path.join(dir, "flame_params.json"), "r") as f:
                params = json.load(f)

            # Get actual image path
            def img_path(dir_name):
                file_name = os.path.basename(frame["file_path"])
                path = os.path.join(dir, dir_name, file_name)
                if os.path.exists(path + ".png"): return path + ".png"
                elif os.path.exists(path + ".jpg"): return path + ".jpg"
                else: assert 0, f"Image {path} does not exist."

            # Load (subsampled) frame data in this subdirectory
            for frame in params["frames"]:
                sub_id = int(os.path.basename(frame["file_path"])) - 1  # start from 0
                assert sub_id >= 0
                if ((sub_id not in self.subsample) if isinstance(self.subsample, list) else
                    (sub_id % self.subsample != 0)):
                    continue
                self.frame_data["sub_id"].append(sub_id)
                self.frame_data["sub_dir"].append(sub_dir)
                self.frame_data["image_path"].append(img_path("image"))
                self.frame_data["mask_path"].append(img_path("mask"))
                if self.use_semantics:
                    self.frame_data["semantic_path"].append(img_path("semantic"))
                world_to_cam = np.array(frame["world_mat"],
                                        dtype=np.float32)  # [3x4] projection mat
                self.frame_data["extrinsic"].append(load_K_Rt_from_P(world_to_cam)[1])
                self.frame_data["expression"].append(np.array(frame["expression"],
                                                              dtype=np.float32))
                self.frame_data["pose"].append(np.array(frame["pose"], dtype=np.float32))
                bbox = (np.array(frame['bbox'], dtype=np.float32) + 1.) * halfsize_bbox
                self.frame_data["bbox"].append(bbox.astype(int))
                keypoints = np.array(frame["flame_keypoints"], dtype=np.float32)
                keypoints = (keypoints + 1) * halfsize_bbox[None, :2]
                self.frame_data["flame_keypoints"].append(keypoints)

        # Load shared shape params and intrinsic matrix
        self.shape_params = np.array(params['shape_params'], dtype=np.float32)
        focal_cxcy = params['intrinsics']
        assert focal_cxcy[3] <= 1
        self.intrinsic = np.array([
            [focal_cxcy[0] * img_res[0], 0, focal_cxcy[2] * img_res[0]],
            [0, focal_cxcy[1] * img_res[1], focal_cxcy[3] * img_res[1]],
            [0, 0, 0],
        ]).astype(np.float32)  # construct intrinsic matrix in pixels

        # Concatenate array list into a single array
        self.frame_data["sub_id"] = np.array(self.frame_data["sub_id"], dtype=np.int32)
        self.frame_data["extrinsic"] = np.stack(self.frame_data["extrinsic"])
        self.frame_data["expression"] = np.stack(self.frame_data["expression"])
        self.frame_data["pose"] = np.stack(self.frame_data["pose"])
        self.frame_data["bbox"] = np.stack(self.frame_data["bbox"])
        self.frame_data["flame_keypoints"] = np.stack(self.frame_data["flame_keypoints"])

    def __len__(self):
        return len(self.frame_data["image_path"])

    def __getitem__(self, idx) -> tuple[dict, dict]:
        uv = torch.meshgrid(torch.arange(self.img_res[0]),
                            torch.arange(self.img_res[1]),
                            indexing="xy")
        uv = torch.stack(uv, -1).float().flatten(0, 1)  # [H*W, 2] corresponds to flattened pixels
        background_rgb = self.background_rgb if self.background_rgb is not None \
            else np.random.rand(3).astype(np.float32)

        # Setup inputs and targets for all ray samples
        inputs = {
            "id": torch.tensor(idx, dtype=torch.long),
            "sub_id": torch.tensor(self.frame_data["sub_id"][idx], dtype=torch.long),
            "sub_dir": self.frame_data["sub_dir"][idx],
            "img_res": torch.tensor(self.img_res, dtype=torch.long),
            "uv": uv,
            "intrinsic": torch.from_numpy(self.intrinsic),
            "extrinsic": torch.from_numpy(self.frame_data["extrinsic"][idx]),
            "expression": torch.from_numpy(self.frame_data["expression"][idx]),
            "pose": torch.from_numpy(self.frame_data["pose"][idx]),
            "background_rgb": torch.from_numpy(background_rgb),
        }

        targets = {}
        if not self.no_gt:
            rgb = load_rgb_image(self.frame_data["image_path"][idx], self.img_res).reshape(-1, 3)
            if self.mask_threshold is not None:
                mask = load_mask(self.frame_data["mask_path"][idx],
                                 self.img_res,
                                 low_threshold=self.mask_threshold).reshape(-1)
            else:
                mask = load_mask(self.frame_data["mask_path"][idx], self.img_res).reshape(-1)
            rgb = rgb * mask[:, None] + background_rgb * (1 - mask[:, None])  # [H*W, 3]
            targets = {
                "rgb": torch.from_numpy(rgb.astype(np.float32)),
                "mask": torch.from_numpy(mask).bool(),
            }
            inputs["mask"] = targets["mask"]
            if self.use_semantics:
                semantic = load_semantic(self.frame_data["semantic_path"][idx], self.img_res)
                targets["semantic"] = torch.from_numpy(semantic.reshape(-1, 9)) >= 1
            targets["keypoints"] = torch.from_numpy(self.frame_data["flame_keypoints"][idx])

        # Sample pixels for the given number of rays
        if self.num_rays != -1:
            assert not self.no_gt, "Cannot sample rays in no_gt mode"

            full_sample_count = self.num_rays // 2
            if self.use_semantics and self.semantic_sampling_ratio > 0:
                semantic_sample_count = self.num_rays // self.semantic_sampling_ratio
                # sample from eye, mouse region and lips
                focus_mask = targets["semantic"][..., 1] + targets["semantic"][..., 3] \
                           + targets["semantic"][..., 4] + targets["semantic"][..., 5] > 0
                uv_focus = uv[focus_mask]
                rgb_focus = targets["rgb"][focus_mask]
                mask_focus = targets["mask"][focus_mask]
                semantic_focus = targets["semantic"][focus_mask]
                sampling_idx_focus = torch.randperm(len(uv_focus))[:semantic_sample_count]
                uv_sample_focus = uv_focus[sampling_idx_focus]
                rgb_sample_focus = rgb_focus[sampling_idx_focus]
                mask_sample_focus = mask_focus[sampling_idx_focus]
                semantic_sample_focus = semantic_focus[sampling_idx_focus]
                full_sample_count -= len(uv_sample_focus)

            # sample half from the pixels from entire image, and half from pixels in the bbox
            sampling_idx = torch.randperm(self.num_pixels)[:full_sample_count]
            uv_sample_full = inputs["uv"][sampling_idx]
            rgb_sample_full = targets["rgb"][sampling_idx]
            mask_sample_full = targets["mask"][sampling_idx]
            if self.use_semantics:
                semantic_sample_full = targets["semantic"][sampling_idx]
            # sample from pixels in the bbox
            bbox = self.frame_data["bbox"][idx]
            bbox_mask = torch.logical_and(torch.logical_and(uv[:, 0] > bbox[0], uv[:, 1] > bbox[1]),
                                          torch.logical_and(uv[:, 0] < bbox[2], uv[:, 1] < bbox[3]))
            uv_bbox = uv[bbox_mask]
            rgb_bbox = targets["rgb"][bbox_mask]
            mask_bbox = targets["mask"][bbox_mask]
            if self.use_semantics:
                semantic_bbox = targets["semantic"][bbox_mask]
            sampling_idx_bbox = torch.randperm(len(uv_bbox))[:self.num_rays - self.num_rays // 2]
            uv_sample_bbox = uv_bbox[sampling_idx_bbox]
            rgb_sample_bbox = rgb_bbox[sampling_idx_bbox]
            mask_sample_bbox = mask_bbox[sampling_idx_bbox]
            if self.use_semantics:
                semantic_sample_bbox = semantic_bbox[sampling_idx_bbox]

            # concatenate samples
            if self.use_semantics and self.semantic_sampling_ratio > 0:
                inputs["uv"] = torch.cat([uv_sample_full, uv_sample_bbox, uv_sample_focus])
                targets["rgb"] = torch.cat([rgb_sample_full, rgb_sample_bbox, rgb_sample_focus])
                inputs["mask"] = targets["mask"] = torch.cat(
                    [mask_sample_full, mask_sample_bbox, mask_sample_focus])
                targets["semantic"] = torch.cat(
                    [semantic_sample_full, semantic_sample_bbox, semantic_sample_focus])
            else:
                inputs["uv"] = torch.cat([uv_sample_full, uv_sample_bbox])
                targets["rgb"] = torch.cat([rgb_sample_full, rgb_sample_bbox])
                inputs["mask"] = targets["mask"] = torch.cat([mask_sample_full, mask_sample_bbox])
                if self.use_semantics:
                    targets["semantic"] = torch.cat([semantic_sample_full, semantic_sample_bbox])

        return inputs, targets

    def get_shape_params(self) -> torch.Tensor:
        return torch.from_numpy(self.shape_params)

    def get_mean_expression(self) -> torch.Tensor:
        return torch.mean(self.get_expression_params(), dim=0, keepdim=True)
    
    def get_std_expression(self) -> torch.Tensor:
        return torch.std(self.get_expression_params(), dim=0, keepdim=True)
    
    def get_mean_pose(self) -> torch.Tensor:
        return torch.mean(self.get_pose_params(), dim=0, keepdim=True)
    
    def get_std_pose(self) -> torch.Tensor:
        return torch.std(self.get_pose_params(), dim=0, keepdim=True)

    def get_expression_params(self) -> torch.Tensor:
        return torch.from_numpy(self.frame_data["expression"])

    def get_pose_params(self) -> torch.Tensor:
        return torch.from_numpy(self.frame_data["pose"])

    def get_extrinsic_params(self) -> torch.Tensor:
        return torch.from_numpy(self.frame_data["extrinsic"])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--sub_dir", type=str, nargs='+', required=True)
    parser.add_argument("--img_res", type=int, default=256)
    parser.add_argument("--num_rays", type=int, default=1024)
    parser.add_argument("--subsample", type=int, default=1)
    parser.add_argument("--use_semantics", action="store_true")
    args = parser.parse_args()

    dataset = RealVideoDataset(args.data_dir, args.sub_dir, (args.img_res, args.img_res),
                               args.num_rays, args.subsample, args.use_semantics)
    print(f"Dataset size: {len(dataset)}")
    inputs, targets = dataset[0]
    print(f"Inputs[0]:")
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor): print(f"  {k}: {v.dtype} {v.shape}\n    {v}")
        else: print(f"  {k}: {v}")
    print(f"Targets[0]:")
    for k, v in targets.items():
        if isinstance(v, torch.Tensor): print(f"  {k}: {v.dtype} {v.shape}\n    {v}")
        else: print(f"  {k}: {v}")
