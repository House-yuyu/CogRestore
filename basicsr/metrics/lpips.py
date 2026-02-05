import torch
import numpy as np
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from basicsr.utils.img_util import img2tensor


def crop_border(imgs, crop_border):
    """Crop borders from images.

    Args:
        imgs (numpy.ndarray or list): Images with shape (H, W, C) or (N, H, W, C).
        crop_border (int): Number of pixels to crop from each side.

    Returns:
        numpy.ndarray: Cropped images.
    """
    if crop_border == 0:
        return imgs

    if isinstance(imgs, list):
        return [crop_border(img, crop_border) for img in imgs]

    if imgs.ndim == 3:  # Single image (H, W, C)
        h, w = imgs.shape[:2]
        return imgs[crop_border:h - crop_border, crop_border:w - crop_border, :]
    elif imgs.ndim == 4:  # Batch (N, H, W, C)
        n, h, w = imgs.shape[:3]
        return imgs[:, crop_border:h - crop_border, crop_border:w - crop_border, :]
    else:
        raise ValueError(f'Unsupported image dimension: {imgs.ndim}')


class _LPIPSMetric:
    """Singleton wrapper for LPIPS model to avoid repeated initialization."""
    _instance = None
    _lpips = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(_LPIPSMetric, cls).__new__(cls)
        return cls._instance

    @classmethod
    def get_lpips(cls, device='cuda'):
        if cls._lpips is None:
            cls._lpips = LearnedPerceptualImagePatchSimilarity(
                net_type='vgg',
                normalize=True,      # expects input in [0, 1], will map to [-1, 1]
                reduction='mean'
            ).to(device).eval()
            # Disable gradient for efficiency
            for param in cls._lpips.parameters():
                param.requires_grad = False
        return cls._lpips

    @classmethod
    def reset(cls):
        """Reset singleton (for testing purposes)."""
        cls._instance = None
        cls._lpips = None


def calculate_lpips(img1, img2, crop_border=0, input_order='HWC',
                    test_y_channel=False, device='cuda'):
    """
    Calculate LPIPS between two images using torchmetrics.

    Args:
        img1 (np.ndarray): Image with shape (H, W, C), range [0, 255], dtype uint8 or float.
        img2 (np.ndarray): Image with shape (H, W, C), range [0, 255], dtype uint8 or float.
        crop_border (int): Crop border pixels before evaluation.
        input_order (str): 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Placeholder for API consistency (not used).
        device (str): Device to run computation on ('cuda' or 'cpu').

    Returns:
        float: LPIPS value (lower is better).
    """
    if img1.shape != img2.shape:
        raise ValueError(f"Image shape mismatch: {img1.shape} vs {img2.shape}")
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"input_order must be 'HWC' or 'CHW', got {input_order}")

    # Convert CHW to HWC if needed
    if input_order == 'CHW':
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)

    # Ensure numpy array
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()

    # Crop border
    if crop_border > 0:
        img1 = crop_border(img1, crop_border)
        img2 = crop_border(img2, crop_border)

    h, w = img1.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size after cropping: {h}x{w}")

    # Normalize to [0, 1] — crucial for LPIPS with normalize=True
    # Assume input is in [0, 255] (standard in BasicSR)
    img1_norm = img1.astype(np.float32) / 255.0
    img2_norm = img2.astype(np.float32) / 255.0

    # Convert to tensor (HWC -> CHW, add batch dim)
    img1_tensor = img2tensor(img1_norm, bgr2rgb=False, float32=True).unsqueeze(0).to(device)
    img2_tensor = img2tensor(img2_norm, bgr2rgb=False, float32=True).unsqueeze(0).to(device)

    # Get LPIPS model
    lpips_model = _LPIPSMetric.get_lpips(device=device)

    # Compute LPIPS
    with torch.no_grad():
        score = lpips_model(img1_tensor, img2_tensor)

    return score.item()