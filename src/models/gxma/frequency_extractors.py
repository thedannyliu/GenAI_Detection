import torch
import numpy as np
from scipy.fftpack import dct
import pywt
from typing import Tuple


def radial_average_spectrum(image: torch.Tensor, num_bins: int = 128) -> torch.Tensor:
    """Compute 1D radial average spectrum of an image."""
    img = image.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.mean(axis=0)
    F = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.abs(F)
    h, w = magnitude.shape
    y, x = np.indices((h, w))
    center = np.array([(h - 1) / 2.0, (w - 1) / 2.0])
    r = np.sqrt((y - center[0]) ** 2 + (x - center[1]) ** 2).flatten()
    magnitude = magnitude.flatten()
    bin_edges = np.linspace(0, r.max(), num_bins + 1)
    radial_sum, _ = np.histogram(r, bins=bin_edges, weights=magnitude)
    counts, _ = np.histogram(r, bins=bin_edges)
    radial_mean = radial_sum / np.maximum(counts, 1)
    return torch.tensor(radial_mean, dtype=torch.float32)


def dct_statistics(image: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """Compute DCT coefficient means for each position across blocks."""
    img = image.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.mean(axis=0)
    h, w = img.shape
    h_pad = (block_size - h % block_size) % block_size
    w_pad = (block_size - w % block_size) % block_size
    img = np.pad(img, ((0, h_pad), (0, w_pad)), mode="reflect")
    blocks = []
    for i in range(0, img.shape[0], block_size):
        for j in range(0, img.shape[1], block_size):
            block = img[i : i + block_size, j : j + block_size]
            coeff = dct(dct(block.T, norm="ortho").T, norm="ortho")
            blocks.append(coeff)
    blocks = np.stack(blocks)
    mean_coeff = blocks.mean(axis=0)
    return torch.tensor(mean_coeff.flatten(), dtype=torch.float32)


def wavelet_statistics(
    image: torch.Tensor, wavelet: str = "haar", level: int = 2, num_bins: int = 64
) -> torch.Tensor:
    """Compute histogram of wavelet coefficients from high-frequency subbands."""
    img = image.detach().cpu().numpy()
    if img.ndim == 3:
        img = img.mean(axis=0)
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    high_coeffs = []
    for lvl in range(1, len(coeffs)):
        high_coeffs.extend(coeffs[lvl])
    coeffs_flat = np.hstack([c.flatten() for c in high_coeffs])
    hist, _ = np.histogram(coeffs_flat, bins=num_bins, density=True)
    return torch.tensor(hist, dtype=torch.float32)


class FrequencyFeatureExtractor:
    """Extract frequency features using multiple methods and concatenate them."""

    def __init__(self) -> None:
        pass

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        f_radial = radial_average_spectrum(image)
        f_dct = dct_statistics(image)
        f_wavelet = wavelet_statistics(image)
        return torch.cat([f_radial, f_dct, f_wavelet], dim=-0)


class FrequencyFeatureExtractorSplit:
    """Extract frequency features and return them separately."""

    def __init__(self) -> None:
        pass

    def __call__(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        f_radial = radial_average_spectrum(image)
        f_dct = dct_statistics(image)
        f_wavelet = wavelet_statistics(image)
        return f_radial, f_dct, f_wavelet
