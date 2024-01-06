import torch
import torchmetrics.functional as metrics
import torchmetrics.image as image_metrics
import face_alignment
import numpy as np
import cv2


class TestMetric(torch.nn.Module):
    def __init__(self, use_gt_mask=True, without_cloth=False, only_face_interior=False):
        super().__init__()
        self.use_gt_mask = use_gt_mask
        self.without_cloth = without_cloth
        self.only_face_interior = only_face_interior
        self.lpips = image_metrics.LearnedPerceptualImagePatchSimilarity(normalize=True)
        try:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False)
        except:
            self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

    def calc_pixel_error(self, rgb_image_output, rgb_image_gt, mask_gt=None):
        """
        Calculate per-pixel error (MSE, MAE, RMSE, L1).
        Args:
            rgb_image_output: (batch_size, 3, H, W). Output RGB image.
            rgb_image_gt: (batch_size, 3, H, W). Ground truth RGB image.
            mask_gt: (batch_size, 1, H, W). Ground truth object mask.
        Returns:
            pixel_error_dict: Dictionary of per-pixel error:
                'mse': (batch_size,). Mean squared error.
                'mae': (batch_size,). Mean absolute error.
                'rmse': (batch_size,). Root mean squared error.
                'L1': (batch_size,). L1 error of color difference.
        """
        pixel_diff = rgb_image_output - rgb_image_gt
        mse_error = torch.square(pixel_diff)
        mae_error = torch.abs(pixel_diff)
        l1_error = torch.norm(pixel_diff, p=1, dim=1, keepdim=True)

        if mask_gt is not None:
            total = mask_gt.float().sum(dim=[1, 2, 3])
            mse_error = mse_error.sum(dim=[1, 2, 3]) / total
            mae_error = mae_error.sum(dim=[1, 2, 3]) / total
            l1_error = l1_error.sum(dim=[1, 2, 3]) / total
        else:
            mse_error = mse_error.mean(dim=[1, 2, 3])
            mae_error = mae_error.mean(dim=[1, 2, 3])
            l1_error = l1_error.mean(dim=[1, 2, 3])

        rmse_error = torch.sqrt(mse_error)

        return {
            'mse': mse_error,
            'mae': mae_error,
            'rmse': rmse_error,
            'L1': l1_error,
        }

    def calc_landmark_error(self, rgb_image_output, keypoints_gt):
        """
        Calculate landmark error regards to ground truth keypoints.
        Args:
            rgb_image_output: (batch_size, 3, H, W). Output RGB image.
            keypoints_gt: (batch_size, 70, 2). Ground truth keypoints.
        Returns:
            keypoints_error: (batch_size,). Mean error of all keypoints.
        """
        keypoints_gt = keypoints_gt[:, :68, :]
        rgb_image_output = torch.clamp(rgb_image_output * 255, 0, 255)
        keypoints_pred_list = self.fa.get_landmarks_from_batch(rgb_image_output)
        keypoints_error = []
        for i in range(keypoints_gt.shape[0]):
            keypoints_pred = torch.from_numpy(keypoints_pred_list[i][:68]).to(keypoints_gt.device)
            error = torch.square(keypoints_pred - keypoints_gt).sum(dim=2).sqrt()
            keypoints_error.append(error)
        return torch.cat(keypoints_error).mean(dim=1)

    def forward(self, outputs, targets):
        batch_size, num_rays, _ = outputs['rgb'].shape
        img_res = outputs['img_res'][0].cpu().tolist()
        is_full_image = img_res[0] * img_res[1] == num_rays
        assert is_full_image, "Metric only works for full image render result"

        rgb_image_output = torch.clamp(outputs['rgb'].detach(), 0.0, 1.0)
        rgb_image_output = rgb_image_output.view(batch_size, *img_res, 3).permute(0, 3, 1, 2)
        rgb_image_gt = targets['rgb'].view(batch_size, *img_res, 3).permute(0, 3, 1, 2)
        mask_gt = targets['mask'].view(batch_size, *img_res, 1).permute(0, 3, 1, 2)

        if self.only_face_interior:
            for batch_idx in range(targets['keypoints'].shape[0]):
                lmks = targets['keypoints'][batch_idx, :68, :].cpu().numpy()
                hull = cv2.convexHull(lmks.astype(np.int32)).squeeze().astype(np.int32)
                mask = np.zeros((img_res[0], img_res[1]), dtype=np.uint8)
                cv2.fillConvexPoly(mask, points=hull, color=1)
                mask = torch.from_numpy(mask).to(mask_gt.device)
                mask_gt[batch_idx] = mask.bool().unsqueeze(0)

        if self.use_gt_mask:
            rgb_image_gt = rgb_image_gt.clone()
            rgb_image_output = rgb_image_output.clone()
            mask_outside = (~mask_gt).expand_as(rgb_image_gt)
            if self.without_cloth:
                semantics = targets['semantic'].view(batch_size, *img_res, -1).permute(0, 3, 1, 2)
                mask_cloth = semantics[:, 7:8].expand_as(rgb_image_gt)
                mask_outside = mask_outside | mask_cloth
            rgb_image_gt[mask_outside] = 0.0
            rgb_image_output[mask_outside] = 0.0

        metric_dict = self.calc_pixel_error(rgb_image_output, rgb_image_gt)
        metric_dict['psnr'] = 20.0 * torch.log10(1.0 / (metric_dict['rmse'] + 1e-8))
        metric_dict['ssim'] = metrics.structural_similarity_index_measure(rgb_image_output,
                                                                          rgb_image_gt,
                                                                          data_range=1.0,
                                                                          reduction='none')
        metric_dict['lpips'] = torch.stack([
            self.lpips(rgb_image_output[i:i + 1], rgb_image_gt[i:i + 1]) for i in range(batch_size)
        ])
        metric_dict['keypoint'] = self.calc_landmark_error(rgb_image_output, targets['keypoints'])

        return metric_dict