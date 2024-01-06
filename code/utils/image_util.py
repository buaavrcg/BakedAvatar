import numpy as np
import imageio
import cv2
import torch
import torchvision


def image_to_float32(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0  # Convert to float32 and normalize to [0, 1]
    else:
        assert img.dtype == np.float32, "Image dtype should be uint8 or float32"
        img = img / 255.0
        assert img.min() >= 0 and img.max() <= 1, f"Values out of range ({img.min(),img.max()})"
    return img


def load_rgb_image(path: str, img_res: tuple[int, int]) -> np.ndarray:
    """
    Args:
        path: Path to the image.
        img_res: Image resolution, e.g. (256, 256)
    Returns:
        img: Normalized image ndarray of shape (img_res[0], img_res[1], 3)
    """
    img = imageio.imread(path)  # HWC
    img = image_to_float32(img)  # Convert to float32 and normalize to [0, 1]
    img = cv2.resize(img, img_res)  # Resize to target resolution
    assert img.shape == (*img_res, 3)
    return img


def load_mask(path: str,
              img_res: tuple[int, int],
              low_threshold: float = 0.5,
              high_threshold: float = 0.99) -> np.ndarray:
    """
    Args:
        path: Path to the mask image.
        img_res: Image resolution, e.g. (256, 256)
        threshold: Alpha value less than threshold will be set to 0.
    Returns:
        mask: (float32) Mask ndarray of shape (img_res[0], img_res[1], 1)
    """
    alpha = imageio.imread(path, mode='L')
    alpha = image_to_float32(alpha)  # Convert to float32 and normalize to [0, 1]
    alpha = cv2.resize(alpha, img_res)  # Resize to target resolution
    alpha = alpha[..., np.newaxis]  # HW -> HW1
    assert alpha.shape == (*img_res, 1)
    alpha[alpha < low_threshold] = 0.0
    alpha[alpha > high_threshold] = 1.0
    # alpha_smooth = alpha[(low_threshold <= alpha) & (alpha <= high_threshold)]
    # alpha_smooth = (alpha_smooth - low_threshold) / (high_threshold - low_threshold)
    # alpha[(low_threshold <= alpha) & (alpha <= high_threshold)] = alpha_smooth
    return alpha


def load_semantic(path: str, img_res: tuple[int, int]) -> np.ndarray:
    """
    Load segmentation from face parsing and combine it to one-hot encoding.
    Args:
        path: Path to the semantic image.
        img_res: Image resolution, e.g. (256, 256)
    Returns:
        semantic: One-hot ndarray of shape (img_res[0], img_res[1], 9)
            channel 0: skin, nose, ears, neck
            channel 1: left eye, right eye
            channel 2: left eyebrow, right eyebrow
            channel 3: mouth interior, eye glasses
            channel 4: upper lip
            channel 5: lower lip
            channel 6: hair
            channel 7: cloth, necklace
            channel 8: background

    Code based on https://github.com/zhengyuf/IMavatar
    """
    img = imageio.imread(path, mode='L')
    semantic = np.zeros((*img.shape, 9), dtype=np.float32)
    semantic[..., 0] = ((img == 1) + (img == 10) + (img == 8) + (img == 7) +
                        (img == 14)) >= 1  # skin, nose, ears, neck
    semantic[..., 1] = ((img == 4) + (img == 5)) >= 1  # left eye, right eye
    semantic[..., 2] = ((img == 2) + (img == 3)) >= 1  # left eyebrow, right eyebrow
    semantic[..., 3] = ((img == 11) + (img == 6)) >= 1  # mouth interior, eye glasses
    semantic[..., 4] = (img == 12)  # upper lip
    semantic[..., 5] = (img == 13)  # lower lip
    semantic[..., 6] = ((img == 17) + (img == 9)) >= 1  # hair
    semantic[..., 7] = ((img == 15) + (img == 16)) >= 1  # cloth, necklace
    semantic[..., 8] = 1. - np.sum(semantic[:, :, :8], 2)  # background
    semantic = cv2.resize(semantic, img_res)  # Resize to target resolution
    return semantic


def visualize_semantic(semantic: np.ndarray) -> np.ndarray:
    """
    Args:
        semantic: One-hot ndarray of shape (H, W, 9)
    Returns:
        color: Colorized semantic image of shape (H, W, 3)

    Code based on https://github.com/zhengyuf/IMavatar
    """
    assert semantic.shape[-1] == 9
    cmap = np.array([
        (204, 0, 0),
        (51, 51, 255),
        (0, 255, 255),
        (102, 204, 0),
        (255, 255, 0),
        (0, 0, 153),
        (0, 0, 204),
        (0, 204, 0),
        (0, 0, 0),
    ])  # [9, 3]
    color = np.matmul(semantic.reshape(-1, 9), cmap)  # [-1, 3]
    return color.reshape(semantic.shape[:-1] + (3, ))  # [H, W, 3]


def visualize_images(img_res,
                     *images,
                     filename=None,
                     normalize=False,
                     value_range=None,
                     padding=1,
                     pad_value=0,
                     nrow=None):
    """
    Visualize a list of tensor images. Each column corresponds to one tensor in images.
    Args:
        img_res: tuple[int, int]. Height and width of images.
        images: list of image tensor of shape (B, H, W, C) or (B, H * W, C). C=1 or C=3.
        filename: str or None. If not None, save the image to this filename.
        normalize: bool. If True, shift all images to the range (0, 1).
        value_range: tuple[float, float]. Min and max used to normalize the images.
        padding: int. Amount of padding. Default: 1.
        pad_value: int. Value for the padded pixels. Default: 0.
        nrow: int. Max number of rows in the grid. Default: len(images).
    Returns:
        grid_tensor: tensor of shape (3, height, width).
    """
    columns = []
    for image in images:
        B, C = image.shape[0], image.shape[-1]
        assert C in [1, 3], f"Invalid image tensor channels ({C})"
        image = image.detach().cpu().view(B, *img_res, C).permute(0, 3, 1, 2)
        columns.append(image.expand(-1, 3, -1, -1))  # expand to 3 channels

    image_tensors = torch.stack(columns, dim=1).flatten(0, 1)
    grid_tensor = torchvision.utils.make_grid(image_tensors,
                                              nrow=(nrow or len(images)),
                                              padding=padding,
                                              normalize=normalize,
                                              value_range=value_range,
                                              pad_value=pad_value)

    if filename is not None:
        torchvision.utils.save_image(grid_tensor, filename)

    return grid_tensor
