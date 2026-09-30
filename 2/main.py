import os
import time
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.transform import rescale

import numpy as np


# ====================================
# Part 1.1: Convolutions from Scratch
# ====================================

def pad(x, k):
    kh, kw = k.shape
    return np.pad(
        x,
        (((kh - 1) // 2, kh // 2), ((kw - 1) // 2, kw // 2)),
        mode="constant",
        constant_values=0,
    )


def conv4(x, k):
    x, k = np.asarray(x, float), np.asarray(k, float)
    padded, flipped = pad(x, k), k[::-1, ::-1]
    out = np.zeros_like(x)
    H, W = x.shape
    kh, kw = k.shape
    for i in range(H):
        for j in range(W):
            for u in range(kh):
                for v in range(kw):
                    out[i, j] += padded[i + u, j + v] * flipped[u, v]
    return out


def conv2(x, k):
    x, k = np.asarray(x, float), np.asarray(k, float)
    padded, flipped = pad(x, k), k[::-1, ::-1]
    out = np.zeros_like(x)
    H, W = x.shape
    kh, kw = k.shape
    for i in range(H):
        for j in range(W):
            out[i, j] = np.sum(padded[i : i + kh, j : j + kw] * flipped)
    return out

def run_part1_1():
    # Make sure media directory exists
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)

    img_name = os.path.join(media_dir, "my_photo.jpg")

    if not os.path.exists(img_name):
        print(f"Warning: '{img_name}' not found.")
        print(
            "Generating a synthetic test gradient image and saving to media/..."
        )
        gray_img = np.tile(np.linspace(0, 1, 250), (250, 1))
        plt.imsave(img_name, gray_img, cmap="gray")
    else:
        raw_img = imread(img_name)
        if raw_img.ndim == 3:
            gray_img = rgb2gray(raw_img)
        else:
            gray_img = raw_img / 255.0 if raw_img.max() > 1.0 else raw_img

    # Rescale down to max dimension 300 so loop operations finish quickly
    if max(gray_img.shape) > 300:
        gray_img = rescale(
            gray_img, 300 / max(gray_img.shape), anti_aliasing=True
        )

    # Define filters
    box = np.ones((9, 9), dtype=np.float64) / 81.0
    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)

    # Compare correctness & timing on a 60x60 patch
    crop = gray_img[:60, :60]

    t0 = time.perf_counter()
    sp_crop = convolve2d(
        crop, box, mode="same", boundary="fill", fillvalue=0.0
    )
    t_sp = time.perf_counter() - t0

    t0 = time.perf_counter()
    c4_crop = conv4(crop, box)
    t_c4 = time.perf_counter() - t0

    t0 = time.perf_counter()
    c2_crop = conv2(crop, box)
    t_c2 = time.perf_counter() - t0

    print("--- Verification & Runtime (60x60 crop) ---")
    print(f"conv4 vs SciPy max diff: {np.max(np.abs(c4_crop - sp_crop)):.2e}")
    print(f"conv2 vs SciPy max diff: {np.max(np.abs(c2_crop - sp_crop)):.2e}")
    print(
        f"Times: SciPy={t_sp*1000:.2f}ms | conv2={t_c2*1000:.2f}ms | conv4={t_c4*1000:.2f}ms"
    )

    # Filter full image with conv2
    img_box = conv2(gray_img, box)
    img_dx = conv2(gray_img, dx)
    img_dy = conv2(gray_img, dy)

    # Save individual JPG outputs directly in media/
    plt.imsave(os.path.join(media_dir, "part1_1_original.jpg"), gray_img, cmap="gray")
    plt.imsave(os.path.join(media_dir, "part1_1_box.jpg"), img_box, cmap="gray")
    plt.imsave(os.path.join(media_dir, "part1_1_dx.jpg"), img_dx, cmap="gray")
    plt.imsave(os.path.join(media_dir, "part1_1_dy.jpg"), img_dy, cmap="gray")

    # Save visual comparison grid in media/
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(gray_img, cmap="gray")
    axes[0].set_title("Original")
    axes[1].imshow(img_box, cmap="gray")
    axes[1].set_title("9x9 Box")
    axes[2].imshow(img_dx, cmap="gray")
    axes[2].set_title("Dx Filter")
    axes[3].imshow(img_dy, cmap="gray")
    axes[3].set_title("Dy Filter")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    grid_path = os.path.join(media_dir, "part1_1_comparison.jpg")
    plt.savefig(grid_path, dpi=200)
    plt.show()

    print(f"All images saved under '{media_dir}/'.")


# =====================================
# Part 1.2: Finite Difference Operator
# =====================================

def run_part1_2():
    print("\n--- Running Part 1.2: Finite Difference Operator ---")
    media_dir = "media"
    cam_name = os.path.join(media_dir, "cameraman.png")

    if not os.path.exists(cam_name):
        from skimage import data

        cam_img = data.camera().astype(np.float64) / 255.0
        plt.imsave(cam_name, cam_img, cmap="gray")
    else:
        raw_cam = imread(cam_name)
        if raw_cam.ndim == 3:
            if raw_cam.shape[2] == 4:
                raw_cam = raw_cam[:, :, :3]
            cam_img = rgb2gray(raw_cam)
        else:
            cam_img = (
                raw_cam / 255.0 if raw_cam.max() > 1.0 else raw_cam.astype(float)
            )

    # Define finite difference operators
    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)

    # Convolve to get partial derivatives
    ix = convolve2d(cam_img, dx, mode="same", boundary="fill", fillvalue=0.0)
    iy = convolve2d(cam_img, dy, mode="same", boundary="fill", fillvalue=0.0)

    # Compute gradient magnitude
    grad_mag = np.sqrt(ix**2 + iy**2)

    # Binarize by tuning a threshold tau
    tau = 0.22
    edge_bin = (grad_mag > tau).astype(np.float64)

    # Normalize derivative images to [0, 1] for saving so negatives aren't clipped to black
    ix_norm = (ix - ix.min()) / (ix.max() - ix.min() + 1e-8)
    iy_norm = (iy - iy.min()) / (iy.max() - iy.min() + 1e-8)

    plt.imsave(os.path.join(media_dir, "part1_2_ix.jpg"), ix_norm, cmap="gray")
    plt.imsave(os.path.join(media_dir, "part1_2_iy.jpg"), iy_norm, cmap="gray")
    plt.imsave(
        os.path.join(media_dir, "part1_2_grad_mag.jpg"), grad_mag, cmap="gray"
    )
    plt.imsave(
        os.path.join(media_dir, "part1_2_binary.jpg"), edge_bin, cmap="gray"
    )

    # Save visual comparison grid
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(ix, cmap="gray")
    axes[0].set_title("Partial Derivative Ix")
    axes[1].imshow(iy, cmap="gray")
    axes[1].set_title("Partial Derivative Iy")
    axes[2].imshow(grad_mag, cmap="gray")
    axes[2].set_title("Gradient Magnitude")
    axes[3].imshow(edge_bin, cmap="gray")
    axes[3].set_title(f"Binarized (tau={tau})")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, "part1_2_comparison.jpg"), dpi=200)
    plt.close()
    print("Part 1.2 complete. Saved outputs to media/.")


# ==============================================
# Part 1.3: Derivative of Gaussian (DoG) Filter
# ==============================================

def run_part1_3():
    print("\n--- Running Part 1.3: Derivative of Gaussian (DoG) Filter ---")
    media_dir = "media"
    cam_name = os.path.join(media_dir, "cameraman.png")

    raw_cam = imread(cam_name)
    if raw_cam.ndim == 3:
        if raw_cam.shape[2] == 4:
            raw_cam = raw_cam[:, :, :3]
        cam_img = rgb2gray(raw_cam)
    else:
        cam_img = (
            raw_cam / 255.0 if raw_cam.max() > 1.0 else raw_cam.astype(float)
        )

    # Construct 2D Gaussian Filter using cv2.getGaussianKernel
    ksize = 9
    sigma = 1.5
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    g_kernel = kernel_1d @ kernel_1d.T

    # Finite difference kernels
    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)

    # Method 1: Blur first, then compute derivatives
    cam_blurred = convolve2d(
        cam_img, g_kernel, mode="same", boundary="fill", fillvalue=0.0
    )
    ix_m1 = convolve2d(
        cam_blurred, dx, mode="same", boundary="fill", fillvalue=0.0
    )
    iy_m1 = convolve2d(
        cam_blurred, dy, mode="same", boundary="fill", fillvalue=0.0
    )
    grad_mag_m1 = np.sqrt(ix_m1**2 + iy_m1**2)
    # Threshold for blurred gradient magnitude
    tau = 0.12
    edge_bin_m1 = (grad_mag_m1 > tau).astype(np.float64)

    # Method 2: Create DoG filters first (Single Convolution)
    dog_x = convolve2d(
        g_kernel, dx, mode="same", boundary="fill", fillvalue=0.0
    )
    dog_y = convolve2d(
        g_kernel, dy, mode="same", boundary="fill", fillvalue=0.0
    )

    ix_m2 = convolve2d(
        cam_img, dog_x, mode="same", boundary="fill", fillvalue=0.0
    )
    iy_m2 = convolve2d(
        cam_img, dog_y, mode="same", boundary="fill", fillvalue=0.0
    )

    print(f"Max absolute difference in Ix (Method 1 vs 2): {np.max(np.abs(ix_m1 - ix_m2)):.2e}")

    grad_mag_m2 = np.sqrt(ix_m2**2 + iy_m2**2)
    edge_bin_m2 = (grad_mag_m2 > tau).astype(np.float64)

    # Verify equivalence
    diff_x = np.max(np.abs(ix_m1 - ix_m2))
    diff_y = np.max(np.abs(iy_m1 - iy_m2))
    print(f"Max absolute difference in Ix (Method 1 vs 2): {diff_x:.2e}")
    print(f"Max absolute difference in Iy (Method 1 vs 2): {diff_y:.2e}")

    # Save Outputs for Webpage
    # Normalize DoG filters for display
    dog_x_disp = (dog_x - dog_x.min()) / (dog_x.max() - dog_x.min() + 1e-8)
    dog_y_disp = (dog_y - dog_y.min()) / (dog_y.max() - dog_y.min() + 1e-8)

    plt.imsave(
        os.path.join(media_dir, "part1_3_blurred.jpg"), cam_blurred, cmap="gray"
    )
    plt.imsave(os.path.join(media_dir, "part1_3_dog_x.jpg"), dog_x_disp, cmap="gray")
    plt.imsave(os.path.join(media_dir, "part1_3_dog_y.jpg"), dog_y_disp, cmap="gray")
    plt.imsave(
        os.path.join(media_dir, "part1_3_grad_mag.jpg"),
        grad_mag_m2,
        cmap="gray",
    )
    plt.imsave(
        os.path.join(media_dir, "part1_3_binary.jpg"), edge_bin_m2, cmap="gray"
    )

    # Plot comparison of DoG filters and final edges
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(dog_x_disp, cmap="gray")
    axes[0].set_title("DoG Filter X")
    axes[1].imshow(dog_y_disp, cmap="gray")
    axes[1].set_title("DoG Filter Y")
    axes[2].imshow(grad_mag_m2, cmap="gray")
    axes[2].set_title("Gradient Mag (DoG)")
    axes[3].imshow(edge_bin_m2, cmap="gray")
    axes[3].set_title(f"Binarized (tau={tau})")

    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, "part1_3_comparison.jpg"), dpi=200)
    plt.close()
    print("Part 1.3 complete. Saved outputs to media/.")


# ==============================================
# Bells & Whistles: Gradient Orientation in HSV
# ==============================================

def compute_gradient_orientation(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    eps = 1e-12  # avoid division by zero
    theta = np.zeros_like(ix)

    pos_x = ix > 0
    neg_x_pos_y = (ix < 0) & (iy >= 0)
    neg_x_neg_y = (ix < 0) & (iy < 0)
    zero_x_pos_y = (np.abs(ix) <= eps) & (iy > 0)
    zero_x_neg_y = (np.abs(ix) <= eps) & (iy < 0)

    # Manual arctan2 logic
    theta[pos_x] = np.arctan(iy[pos_x] / (ix[pos_x] + eps))
    theta[neg_x_pos_y] = np.arctan(iy[neg_x_pos_y] / (ix[neg_x_pos_y] - eps)) + np.pi
    theta[neg_x_neg_y] = np.arctan(iy[neg_x_neg_y] / (ix[neg_x_neg_y] - eps)) - np.pi
    theta[zero_x_pos_y] = np.pi / 2.0
    theta[zero_x_neg_y] = -np.pi / 2.0

    return theta


def run_part1_bells():
    print("\n--- Running Part 1 Bells & Whistles: Gradient Orientation Visualization ---")
    media_dir = "media"
    cam_name = os.path.join(media_dir, "cameraman.png")

    raw_cam = plt.imread(cam_name)
    if raw_cam.dtype == np.uint8 or raw_cam.max() > 1.0:
        cam_img = raw_cam.astype(np.float64) / 255.0
    else:
        cam_img = raw_cam.astype(np.float64)

    if cam_img.ndim == 3:
        if cam_img.shape[2] == 4:
            cam_img = cam_img[..., :3]
        cam_img = (
            0.299 * cam_img[..., 0]
            + 0.587 * cam_img[..., 1]
            + 0.114 * cam_img[..., 2]
        )

    # Use smoothed Gaussian derivatives (DoG) for cleaner orientations
    ksize = 9
    sigma = 1.5
    k1d = cv2.getGaussianKernel(ksize, sigma)
    g_kernel = k1d @ k1d.T
    cam_blur = convolve2d(cam_img, g_kernel, mode="same", boundary="fill", fillvalue=0.0)

    dx = np.array([[1, -1]], dtype=np.float64)
    dy = np.array([[1], [-1]], dtype=np.float64)
    ix = convolve2d(cam_blur, dx, mode="same", boundary="fill", fillvalue=0.0)
    iy = convolve2d(cam_blur, dy, mode="same", boundary="fill", fillvalue=0.0)

    # Gradient magnitude & custom orientation
    grad_mag = np.sqrt(ix**2 + iy**2)
    theta = compute_gradient_orientation(ix, iy)

    # Construct HSV Image
    # Hue: map [-pi, pi] to [0, 1]
    H = (theta + np.pi) / (2 * np.pi)
    
    # Saturation: full saturation
    S = np.ones_like(H)
    
    # Scale by gradient magnitude so flat background is black
    norm_mag = grad_mag / (grad_mag.max() + 1e-8)
    # Boost contrast with a power curve
    V = np.clip(norm_mag * 2.5, 0.0, 1.0)

    hsv_img = np.stack([H, S, V], axis=-1)
    rgb_orientation = hsv_to_rgb(hsv_img)

    # Save results
    plt.imsave(os.path.join(media_dir, "part1_bells_hsv_orientation.jpg"), rgb_orientation)

    # Plot figure with colorwheel explanation
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(cam_img, cmap="gray")
    axes[0].set_title("Original Image")
    axes[1].imshow(grad_mag, cmap="gray")
    axes[1].set_title("Gradient Magnitude")
    im = axes[2].imshow(rgb_orientation)
    axes[2].set_title("Orientation (Hue) * Magnitude (Value)")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, "part1_bells_orientation_comparison.jpg"), dpi=200)
    plt.close()
    print("Part 1 Bells & Whistles complete. Saved outputs to media/.")


# =============================
# Part 2.1: Image "Sharpening"
# =============================

def load_image(filepath: str) -> np.ndarray:
    """Loads an image normalized to float [0, 1] handling RGB, RGBA, or Grayscale."""
    raw = plt.imread(filepath)
    if raw.dtype == np.uint8 or raw.max() > 1.0:
        img = raw.astype(np.float64) / 255.0
    else:
        img = raw.astype(np.float64)
    # Strip alpha channel if present
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    return img

def filter_channel(channel: np.ndarray, g_kernel: np.ndarray, alpha: float) -> tuple:
    """Sharpens a single 2D channel."""
    blurred = convolve2d(channel, g_kernel, mode="same", boundary="symm")
    high_freq = channel - blurred
    sharpened = channel + alpha * high_freq
    return blurred, high_freq, np.clip(sharpened, 0.0, 1.0)

def unsharp_mask(image: np.ndarray, sigma: float = 1.5, alpha: float = 1.0) -> tuple:
    """Applies unsharp masking to an image (supports 2D grayscale and 3D RGB)."""
    ksize = int(np.ceil(sigma * 6)) | 1
    k1d = cv2.getGaussianKernel(ksize, sigma)
    g_kernel = k1d @ k1d.T

    if image.ndim == 3:
        H, W, C = image.shape
        blurred = np.zeros((H, W, C), dtype=np.float64)
        high_freq = np.zeros((H, W, C), dtype=np.float64)
        sharpened = np.zeros((H, W, C), dtype=np.float64)
        for c in range(C):
            b, h, s = filter_channel(image[..., c], g_kernel, alpha)
            blurred[..., c] = b
            high_freq[..., c] = h
            sharpened[..., c] = s
        return blurred, high_freq, sharpened
    else:
        return filter_channel(image, g_kernel, alpha)

def run_part2_1():
    print('\n--- Running Part 2.1: Image "Sharpening" ---')
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)

    # 1. Multi-Image Sharpening
    test_cases = [
        {   "file": "taj.jpg", 
            "sigma": 1.5, 
            "alpha": 1.5, 
            "name": "taj"},
        {
            "file": "library.jpg",
            "sigma": 1.5,
            "alpha": 1.5,
            "name": "library",
        },
        {
            "file": "blocks.jpg",
            "sigma": 1.5,
            "alpha": 1.5,
            "name": "blocks",
        },
    ]

    for item in test_cases:
        path = os.path.join(media_dir, item["file"])
        if not os.path.exists(path):
            print(f"Skipping '{item['file']}': file not found in {media_dir}/.")
            continue

        img = load_image(path)
        sigma = item["sigma"]
        alpha = item["alpha"]
        prefix = item["name"]
        cmap = "gray" if img.ndim == 2 else None

        blurred, high_freq, sharpened = unsharp_mask(
            img, sigma=sigma, alpha=alpha
        )

        # High frequency visualization: shift by +0.5 so negatives are visible as mid-gray
        high_freq_disp = np.clip(high_freq + 0.5, 0.0, 1.0)

        # Save individual images
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_orig.jpg"),
            img,
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_blur.jpg"),
            blurred,
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_high.jpg"),
            high_freq_disp,
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_sharp.jpg"),
            sharpened,
            cmap=cmap,
        )

        # Save comparison grid
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(img, cmap=cmap)
        axes[0].set_title(f"Original ({prefix})")
        axes[1].imshow(blurred, cmap=cmap)
        axes[1].set_title(f"Gaussian Blur (σ={sigma})")
        axes[2].imshow(high_freq_disp, cmap=cmap)
        axes[2].set_title("High Frequencies (shifted +0.5)")
        axes[3].imshow(sharpened, cmap=cmap)
        axes[3].set_title(f"Sharpened (α={alpha})")

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(media_dir, f"part2_1_{prefix}_grid.jpg"), dpi=200
        )
        plt.close()
        print(f"Saved results for {prefix}.")

    # 2. Alpha Parameter Sweep 
    sweep_target = "taj.jpg"
    sweep_path = os.path.join(media_dir, sweep_target)
    if os.path.exists(sweep_path):
        img_sweep = load_image(sweep_path)
        cmap = "gray" if img_sweep.ndim == 2 else None
        alphas = [0.5, 1.0, 2.5, 5.0]

        fig, axes = plt.subplots(1, len(alphas), figsize=(20, 5))
        for idx, a in enumerate(alphas):
            _, _, sharp = unsharp_mask(img_sweep, sigma=1.5, alpha=a)
            plt.imsave(
                os.path.join(media_dir, f"part2_1_taj_alpha_{a}.jpg"),
                sharp,
                cmap=cmap,
            )
            axes[idx].imshow(sharp, cmap=cmap)
            axes[idx].set_title(f"α = {a}")
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig(
            os.path.join(media_dir, "part2_1_alpha_sweep_grid.jpg"), dpi=200
        )
        plt.close()
        print("Saved alpha sweep comparison.")

    # 3. Evaluation: Sharp to Blur, then Re-Sharpen
    eval_target = (
        "library.jpg"
        if os.path.exists(os.path.join(media_dir, "library.jpg"))
        else "blocks.jpg"
    )
    eval_path = os.path.join(media_dir, eval_target)

    if os.path.exists(eval_path):
        sharp_orig = load_image(eval_path)
        cmap = "gray" if sharp_orig.ndim == 2 else None

        # Artificially blur with Gaussian
        ksize = 11
        k1d = cv2.getGaussianKernel(ksize, sigma=2.0)
        g_kernel = k1d @ k1d.T

        if sharp_orig.ndim == 3:
            art_blurred = np.zeros_like(sharp_orig)
            for c in range(sharp_orig.shape[2]):
                art_blurred[..., c] = convolve2d(
                    sharp_orig[..., c], g_kernel, mode="same", boundary="symm"
                )
        else:
            art_blurred = convolve2d(
                sharp_orig, g_kernel, mode="same", boundary="symm"
            )

        # Resharpen the blurred image
        _, _, re_sharpened = unsharp_mask(art_blurred, sigma=2.0, alpha=2.5)

        # Save evaluation outputs
        plt.imsave(
            os.path.join(media_dir, "part2_1_eval_orig.jpg"),
            np.clip(sharp_orig, 0.0, 1.0),
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, "part2_1_eval_blurred.jpg"),
            np.clip(art_blurred, 0.0, 1.0),
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, "part2_1_eval_resharp.jpg"),
            np.clip(re_sharpened, 0.0, 1.0),
            cmap=cmap,
        )

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(sharp_orig, cmap=cmap)
        axes[0].set_title("1. Original Sharp")
        axes[1].imshow(art_blurred, cmap=cmap)
        axes[1].set_title("2. Artificially Blurred")
        axes[2].imshow(re_sharpened, cmap=cmap)
        axes[2].set_title("3. Re-Sharpened")

        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(
            os.path.join(media_dir, "part2_1_eval_comparison.jpg"), dpi=200
        )
        plt.close()
        print("Saved evaluation comparison (sharp -> blur -> re-sharpen).")

    print("Part 2.1 complete!")

if __name__ == "__main__":
    # run_part1_1()
    # run_part1_2()
    # run_part1_3()
    # run_part1_bells()
    run_part2_1()
