import os
import time
import cv2
import numpy as np

import matplotlib
try:
    matplotlib.use("TkAgg")
except Exception:
    try:
        matplotlib.use("MacOSX")
    except Exception:
        pass

import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
from scipy.signal import convolve2d
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.transform import rescale
from align_image_code import align_images

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
            "sigma": 2.5, 
            "alpha": 2.0, 
            "name": "taj"},
        {
            "file": "library.jpg",
            "sigma": 3.0,
            "alpha": 2.5,
            "name": "library",
        },
        {
            "file": "blocks.jpg",
            "sigma": 3.0,
            "alpha": 2.5,
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
        high_freq_vis = np.clip(0.5 + high_freq * 3.5, 0.0, 1.0)

        # Save individual images
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_orig.jpg"),
            np.clip(img, 0.0, 1.0),
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_blur.jpg"),
            np.clip(blurred, 0.0, 1.0),
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_high.jpg"),
            high_freq_vis,
            cmap=cmap,
        )
        plt.imsave(
            os.path.join(media_dir, f"part2_1_{prefix}_sharp.jpg"),
            np.clip(sharpened, 0.0, 1.0),
            cmap=cmap,
        )

        # Save comparison grid
        fig, axes = plt.subplots(1, 4, figsize=(18, 5))
        axes[0].imshow(img, cmap=cmap)
        axes[0].set_title(f"Original ({prefix})")
        axes[1].imshow(blurred, cmap=cmap)
        axes[1].set_title(f"Gaussian Blur (σ={sigma})")
        axes[2].imshow(high_freq_vis, cmap=cmap)
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
        ksize = 19
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

        art_blurred = np.clip(art_blurred, 0.0, 1.0)

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

import align_image_code as aic


# ========================
# Part 2.2: Hybrid Images
# ========================

def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    ksize = int(np.ceil(sigma * 6)) | 1
    return cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)

def gaussian_blur_2d(channel: np.ndarray, sigma: float) -> np.ndarray:
    return gaussian_blur(channel, sigma)

def make_hybrid(
    im_low: np.ndarray,
    im_high: np.ndarray,
    sigma_low: float,
    sigma_high: float,
) -> tuple:
    """Combines low frequencies of im_low with high frequencies of im_high."""
    low_freq = gaussian_blur(im_low, sigma_low)
    high_freq = im_high - gaussian_blur(im_high, sigma_high)
    hybrid = low_freq + high_freq
    return low_freq, high_freq, np.clip(hybrid, 0.0, 1.0)

def fourier_analysis(img: np.ndarray) -> np.ndarray:
    """Computes the log magnitude 2D FFT for frequency spectrum analysis."""
    if img.ndim == 3:
        gray = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
    else:
        gray = img
    f_shift = np.fft.fftshift(np.fft.fft2(gray))
    return np.log(np.abs(f_shift) + 1e-8)

def process_hybrid_pair(pair_config: dict, media_dir: str = "media"):
    name = pair_config["name"]
    print(f"\n===========================")
    print(f"Processing Pair: {name}")
    print(f"===========================")

    p1_path = os.path.join(media_dir, pair_config["im1_file"])
    p2_path = os.path.join(media_dir, pair_config["im2_file"])

    if not os.path.exists(p1_path) or not os.path.exists(p2_path):
        print(f"Error: Could not find '{p1_path}' or '{p2_path}'. Skipping.")
        return

    raw1 = load_image(p1_path)
    raw2 = load_image(p2_path)

    # Convert to grayscale
    im1 = 0.299 * raw1[..., 0] + 0.587 * raw1[..., 1] + 0.114 * raw1[..., 2] if raw1.ndim == 3 else raw1.copy()
    im2 = 0.299 * raw2[..., 0] + 0.587 * raw2[..., 1] + 0.114 * raw2[..., 2] if raw2.ndim == 3 else raw2.copy()

    # 1. Interactive Alignment
    print("\n>>> INSTRUCTIONS FOR CLICKING <<<")
    print(f"Window 1 ({pair_config['im1_label']}): Click Point 1, then Point 2.")
    print(f"Window 2 ({pair_config['im2_label']}): Click Point 1, then Point 2 in the SAME order.")
    
    # im1 gets rotated/scaled to align onto im2
    im1_aligned, im2_aligned = align_images(im1, im2)

    # 2. Crop to clear zero-padding borders
    h, w = im1_aligned.shape[:2]
    crop_t = int(pair_config.get("crop_top", 0.15) * h)
    crop_b = int(pair_config.get("crop_bottom", 0.85) * h)
    crop_l = int(pair_config.get("crop_left", 0.12) * w)
    crop_r = int(pair_config.get("crop_right", 0.88) * w)

    im1_c = im1_aligned[crop_t:crop_b, crop_l:crop_r]
    im2_c = im2_aligned[crop_t:crop_b, crop_l:crop_r]

    # 3. Route High/Low Frequency Sources
    sigma_low = pair_config["sigma_low"]
    sigma_high = pair_config["sigma_high"]

    # Check whether im1 or im2 provides high frequencies
    if pair_config["high_source"] == "im1":
        im_high, im_low = im1_c, im2_c
    else:
        im_high, im_low = im2_c, im1_c

    low_f = gaussian_blur(im_low, sigma_low)
    high_f = im_high - gaussian_blur(im_high, sigma_high)
    hybrid = np.clip(low_f + high_f, 0.0, 1.0)

    # 4. Save Image Deliverables
    high_f_vis = np.clip(0.5 + high_f * 3.0, 0.0, 1.0)
    plt.imsave(os.path.join(media_dir, f"part2_2_{name}_hybrid.jpg"), hybrid, cmap="gray")
    plt.imsave(os.path.join(media_dir, f"part2_2_{name}_high.jpg"), high_f_vis, cmap="gray")
    plt.imsave(os.path.join(media_dir, f"part2_2_{name}_low.jpg"), np.clip(low_f, 0.0, 1.0), cmap="gray")

    # 5. Fourier Analysis Grid (Crucial for writeup)
    fft_im1 = fourier_analysis(im1_c)
    fft_im2 = fourier_analysis(im2_c)
    fft_high = fourier_analysis(high_f)
    fft_low = fourier_analysis(low_f)
    fft_hy = fourier_analysis(hybrid)

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes[0, 0].imshow(im1_c, cmap="gray"); axes[0, 0].set_title(pair_config["im1_label"])
    axes[0, 1].imshow(im2_c, cmap="gray"); axes[0, 1].set_title(pair_config["im2_label"])
    axes[0, 2].imshow(high_f_vis, cmap="gray"); axes[0, 2].set_title("High-Pass Filtered")
    axes[0, 3].imshow(low_f, cmap="gray"); axes[0, 3].set_title("Low-Pass Filtered")
    axes[0, 4].imshow(hybrid, cmap="gray"); axes[0, 4].set_title("Hybrid Image")

    axes[1, 0].imshow(fft_im1, cmap="magma"); axes[1, 0].set_title("FFT: Input 1")
    axes[1, 1].imshow(fft_im2, cmap="magma"); axes[1, 1].set_title("FFT: Input 2")
    axes[1, 2].imshow(fft_high, cmap="magma"); axes[1, 2].set_title("FFT: High-Pass")
    axes[1, 3].imshow(fft_low, cmap="magma"); axes[1, 3].set_title("FFT: Low-Pass")
    axes[1, 4].imshow(fft_hy, cmap="magma"); axes[1, 4].set_title("FFT: Hybrid")

    for row in axes:
        for ax in row:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(media_dir, f"part2_2_{name}_fft_grid.jpg"), dpi=200)
    plt.close()
    print(f"Finished {name}! Outputs saved to {media_dir}/.")


def run_part2_2():
    print("\n--- Running Part 2.2: Hybrid Images ---")
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)

    pairs = [
        # Pair 1: Derek + Nutmeg (Nutmeg provides high SF; Derek is anchor im2)
        {
            "name": "derek_nutmeg",
            "im1_file": "nutmeg.jpg",
            "im2_file": "DerekPicture.jpg",
            "im1_label": "Nutmeg (High SF)",
            "im2_label": "Derek (Low SF)",
            "high_source": "im1",
            "sigma_low": 9.0,
            "sigma_high": 3.5,
            "crop_top": 0.18,
            "crop_bottom": 0.85,
            "crop_left": 0.12,
            "crop_right": 0.88,
        },
        # Pair 2: 
        {
            "name": "me_bear",
            "im1_file": "bear.jpg",      # Image to be scaled/rotated
            "im2_file": "my_photo.jpg",      # Target upright anchor
            "im1_label": "Bear (High SF)",
            "im2_label": "Me (Low SF)",
            "high_source": "im1",                # "im1" or "im2" for high frequencies
            "sigma_low": 8.0,
            "sigma_high": 2.5,
            "crop_top": 0.10,
            "crop_bottom": 0.90,
            "crop_left": 0.10,
            "crop_right": 0.90,
        },
        # Pair 3:
        {
            "name": "milk_icecream",
            "im1_file": "milk.jpg",
            "im2_file": "ice_cream.jpg",
            "im1_label": "Milk (High SF)",
            "im2_label": "Ice Cream (Low SF)",
            "high_source": "im1",
            "sigma_low": 8.0,
            "sigma_high": 3.0,
            "crop_top": 0.10,
            "crop_bottom": 0.90,
            "crop_left": 0.10,
            "crop_right": 0.90,
        },
    ]

    for pair in pairs:
        process_hybrid_pair(pair, media_dir=media_dir)

# ===============================
# Bells & Whistles: Using Colors
# ===============================

def make_hybrid_color(
    im_low: np.ndarray,
    im_high: np.ndarray,
    sigma_low: float,
    sigma_high: float,
    color_mode: str = "low",
) -> np.ndarray:
    low_f = gaussian_blur(im_low, sigma_low)
    high_f = im_high - gaussian_blur(im_high, sigma_high)

    def to_gray(img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        return img

    if low_f.ndim == 2:
        low_f = np.repeat(low_f[:, :, np.newaxis], 3, axis=2)
    if high_f.ndim == 2:
        high_f = np.repeat(high_f[:, :, np.newaxis], 3, axis=2)

    if color_mode == "low":
        high_gray = to_gray(high_f)[:, :, np.newaxis]
        hybrid = low_f + high_gray
    elif color_mode == "high":
        low_gray = to_gray(low_f)[:, :, np.newaxis]
        hybrid = low_gray + high_f
    elif color_mode == "both":
        hybrid = low_f + high_f
    else:
        raise ValueError("Invalid color_mode. Choose 'low', 'high', or 'both'.")

    return np.clip(hybrid, 0.0, 1.0)


def test_color_hybrids():
    print("\n--- Running Part 2.2 Bells & Whistles: Color Analysis ---")
    media_dir = "media"
    derek_path = os.path.join(media_dir, "DerekPicture.jpg")
    nutmeg_path = os.path.join(media_dir, "nutmeg.jpg")

    derek_rgb = load_image(derek_path)
    nutmeg_rgb = load_image(nutmeg_path)

    print("\nSelect pupil landmarks for color alignment:")
    print("Window 1 (Nutmeg): Click LEFT pupil, then RIGHT pupil.")
    print("Window 2 (Derek):  Click LEFT pupil, then RIGHT pupil.")
    nutmeg_aligned, derek_aligned = align_images(nutmeg_rgb, derek_rgb)

    h, w = nutmeg_aligned.shape[:2]
    crop_t = int(0.18 * h)
    crop_b = int(0.85 * h)
    crop_l = int(0.12 * w)
    crop_r = int(0.88 * w)

    nutmeg_c = nutmeg_aligned[crop_t:crop_b, crop_l:crop_r]
    derek_c = derek_aligned[crop_t:crop_b, crop_l:crop_r]

    sigma_low = 10.0
    sigma_high = 3.5

    print("Generating color combinations...")
    hybrid_low = make_hybrid_color(derek_c, nutmeg_c, sigma_low, sigma_high, "low")
    hybrid_high = make_hybrid_color(derek_c, nutmeg_c, sigma_low, sigma_high, "high")
    hybrid_both = make_hybrid_color(derek_c, nutmeg_c, sigma_low, sigma_high, "both")

    plt.imsave(os.path.join(media_dir, "part2_2_color_low_only.jpg"), hybrid_low)
    plt.imsave(os.path.join(media_dir, "part2_2_color_high_only.jpg"), hybrid_high)
    plt.imsave(os.path.join(media_dir, "part2_2_color_both.jpg"), hybrid_both)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(hybrid_low)
    axes[0].set_title("1. Color in Low SF Only\n(Derek in color, Nutmeg gray)", fontsize=13)
    axes[1].imshow(hybrid_high)
    axes[1].set_title("2. Color in High SF Only\n(Nutmeg in color, Derek gray)", fontsize=13)
    axes[2].imshow(hybrid_both)
    axes[2].set_title("3. Color in Both SF\n(Both components in color)", fontsize=13)

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    comparison_out = os.path.join(media_dir, "part2_2_color_comparison_grid.jpg")
    plt.savefig(comparison_out, dpi=200)
    plt.close()
    print(f"Color testing complete! Check '{comparison_out}' in media/.")


# ========================================
# Part 2.3: Gaussian and Laplacian Stacks
# ========================================

def build_gaussian_stack(
    img: np.ndarray, num_levels: int = 5, start_sigma: float = 2.0
) -> list:
    """Builds a Gaussian Stack where each level stays at the original full resolution.

    Level 0 is the original image, with sigma doubling at each subsequent level.
    """
    g_stack = [img.copy()]
    curr_sigma = start_sigma
    for i in range(num_levels - 1):
        blurred = gaussian_blur(img, curr_sigma)
        g_stack.append(blurred)
        curr_sigma *= 2.0
    return g_stack

def build_laplacian_stack(g_stack: list) -> list:
    """Builds a Laplacian Stack from a Gaussian Stack.

    L_i = G_i - G_{i+1} for all levels except the last, which retains G_last.
    """
    l_stack = []
    num_levels = len(g_stack)
    for i in range(num_levels - 1):
        l_stack.append(g_stack[i] - g_stack[i + 1])
    # The lowest frequency residual is the final level of the Gaussian stack
    l_stack.append(g_stack[-1].copy())
    return l_stack

def reconstruct_from_laplacian(l_stack: list) -> np.ndarray:
    """Reconstructs the original image by summing all levels of the Laplacian stack."""
    reconstructed = np.zeros_like(l_stack[0])
    for level in l_stack:
        reconstructed += level
    return np.clip(reconstructed, 0.0, 1.0)

# Visualization Helpers
def plot_stack(stack: list, title: str, filepath: str, is_laplacian: bool = False):
    """Plots all levels of a stack side-by-side."""
    num_levels = len(stack)
    fig, axes = plt.subplots(1, num_levels, figsize=(3.2 * num_levels, 3.5))

    for i in range(num_levels):
        level = stack[i]
        if is_laplacian and i < num_levels - 1:
            # Shift mean to 0.5 and boost contrast for visualization of zero-centered band-pass details
            disp = np.clip(0.5 + level * 2.5, 0.0, 1.0)
        else:
            disp = np.clip(level, 0.0, 1.0)

        if disp.ndim == 2:
            axes[i].imshow(disp, cmap="gray")
        else:
            axes[i].imshow(disp)

        axes[i].set_title(f"Level {i}")
        axes[i].axis("off")

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

def plot_burt_adelson_figure10(
    l1_stack: list,
    l2_stack: list,
    name1: str,
    name2: str,
    filepath: str,
):
    
    num_levels = len(l1_stack)
    fig, axes = plt.subplots(2, num_levels, figsize=(3.2 * num_levels, 6.5))

    for i in range(num_levels):
        # Image 1 (Row 0)
        level1 = l1_stack[i]
        disp1 = np.clip(0.5 + level1 * 2.5, 0.0, 1.0) if i < num_levels - 1 else np.clip(level1, 0.0, 1.0)
        if disp1.ndim == 2:
            axes[0, i].imshow(disp1, cmap="gray")
        else:
            axes[0, i].imshow(disp1)
        axes[0, i].set_title(f"{name1} L{i}")
        axes[0, i].axis("off")

        # Image 2 (Row 1)
        level2 = l2_stack[i]
        disp2 = np.clip(0.5 + level2 * 2.5, 0.0, 1.0) if i < num_levels - 1 else np.clip(level2, 0.0, 1.0)
        if disp2.ndim == 2:
            axes[1, i].imshow(disp2, cmap="gray")
        else:
            axes[1, i].imshow(disp2)
        axes[1, i].set_title(f"{name2} L{i}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

# Part 2.3 Runner
def run_part2_3():
    print("\n--- Running Part 2.3: Gaussian & Laplacian Stacks ---")
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)

    apple_path = os.path.join(media_dir, "apple.jpeg")
    orange_path = os.path.join(media_dir, "orange.jpeg")

    if not os.path.exists(apple_path) or not os.path.exists(orange_path):
        print(f"Error: Ensure 'apple.jpeg' and 'orange.jpeg' exist in {media_dir}/.")
        return

    # 1. Load images
    apple = load_image(apple_path)
    orange = load_image(orange_path)

    # 2. Crop to common minimum dimensions
    min_h = min(apple.shape[0], orange.shape[0])
    min_w = min(apple.shape[1], orange.shape[1])
    apple = apple[:min_h, :min_w]
    orange = orange[:min_h, :min_w]

    num_levels = 5
    start_sigma = 2.0

    # 3. Build Stacks for Apple
    print("Building Gaussian and Laplacian stacks for Apple...")
    apple_g = build_gaussian_stack(apple, num_levels=num_levels, start_sigma=start_sigma)
    apple_l = build_laplacian_stack(apple_g)

    # 4. Build Stacks for Orange
    print("Building Gaussian and Laplacian stacks for Orange...")
    orange_g = build_gaussian_stack(orange, num_levels=num_levels, start_sigma=start_sigma)
    orange_l = build_laplacian_stack(orange_g)

    # 5. Sanity Check
    apple_recon = reconstruct_from_laplacian(apple_l)
    recon_error = np.max(np.abs(apple - apple_recon))
    print(f"Reconstruction Sanity Check -> Max absolute difference: {recon_error:.2e}")

    # 6. Save Stack Visualizations
    plot_stack(
        apple_g,
        "Apple: Gaussian Stack",
        os.path.join(media_dir, "part2_3_apple_gaussian_stack.jpg"),
        is_laplacian=False,
    )
    plot_stack(
        apple_l,
        "Apple: Laplacian Stack",
        os.path.join(media_dir, "part2_3_apple_laplacian_stack.jpg"),
        is_laplacian=True,
    )

    plot_stack(
        orange_g,
        "Orange: Gaussian Stack",
        os.path.join(media_dir, "part2_3_orange_gaussian_stack.jpg"),
        is_laplacian=False,
    )
    plot_stack(
        orange_l,
        "Orange: Laplacian Stack",
        os.path.join(media_dir, "part2_3_orange_laplacian_stack.jpg"),
        is_laplacian=True,
    )

    # 7. Recreate Figure 10 from Burt & Adelson
    plot_burt_adelson_figure10(
        apple_l,
        orange_l,
        "Apple",
        "Orange",
        os.path.join(media_dir, "part2_3_figure10_laplacian_comparison.jpg"),
    )

    print("Part 2.3 complete! All stack figures saved to media/.")


# =======================================================
# Part 2.4: Multiresolution Blending (a.k.a. the oraple!)
# =======================================================

def multiresolution_blend(
    im1: np.ndarray,
    im2: np.ndarray,
    mask: np.ndarray,
    num_levels: int = 5,
    start_sigma: float = 2.0,
) -> tuple:
    """Performs multiresolution blending using Gaussian and Laplacian stacks.
    
    Blend equation at each level i:
        L_blend[i] = G_mask[i] * L1[i] + (1 - G_mask[i]) * L2[i]
    """
    # 1. Match dimensions
    h, w = im1.shape[:2]
    im2 = im2[:h, :w]
    mask = mask[:h, :w]

    # 2. Build Laplacian stacks for images
    g1 = build_gaussian_stack(im1, num_levels=num_levels, start_sigma=start_sigma)
    l1 = build_laplacian_stack(g1)

    g2 = build_gaussian_stack(im2, num_levels=num_levels, start_sigma=start_sigma)
    l2 = build_laplacian_stack(g2)

    # 3. Build Gaussian stack for the mask
    gm = build_gaussian_stack(mask, num_levels=num_levels, start_sigma=start_sigma)

    # 4. Blend level by level
    blended_l_stack = []
    l1_masked_stack = []
    l2_masked_stack = []

    for i in range(num_levels):
        m_level = gm[i]
        # Broadcast mask across RGB channels if necessary
        if im1.ndim == 3 and m_level.ndim == 2:
            m_level = m_level[..., np.newaxis]

        term1 = m_level * l1[i]
        term2 = (1.0 - m_level) * l2[i]
        blend_level = term1 + term2

        l1_masked_stack.append(term1)
        l2_masked_stack.append(term2)
        blended_l_stack.append(blend_level)

    # 5. Reconstruct final blended image
    blended_image = reconstruct_from_laplacian(blended_l_stack)
    return blended_image, blended_l_stack, l1_masked_stack, l2_masked_stack, gm

def plot_blending_process(
    l1_masked: list,
    l2_masked: list,
    blended_l: list,
    final_img: np.ndarray,
    label1: str,
    label2: str,
    filepath: str,
):
    """Plots a 3-row grid showing:
      Row 0: Masked Laplacian levels of Image 1
      Row 1: Masked Laplacian levels of Image 2
      Row 2: Combined Blended Laplacian levels
    """
    num_levels = len(blended_l)
    fig, axes = plt.subplots(3, num_levels, figsize=(3.2 * num_levels, 9.5))

    for i in range(num_levels):
        # Scale intermediate bandpass levels around 0.5 for clear visual display
        def prepare_disp(img, is_last):
            if is_last:
                return np.clip(img, 0.0, 1.0)
            return np.clip(0.5 + img * 2.5, 0.0, 1.0)

        is_last = (i == num_levels - 1)
        disp1 = prepare_disp(l1_masked[i], is_last)
        disp2 = prepare_disp(l2_masked[i], is_last)
        disp_b = prepare_disp(blended_l[i], is_last)

        axes[0, i].imshow(disp1 if disp1.ndim == 3 else disp1, cmap="gray" if disp1.ndim == 2 else None)
        axes[0, i].set_title(f"{label1} * Mask (L{i})")
        axes[0, i].axis("off")

        axes[1, i].imshow(disp2 if disp2.ndim == 3 else disp2, cmap="gray" if disp2.ndim == 2 else None)
        axes[1, i].set_title(f"{label2} * (1-Mask) (L{i})")
        axes[1, i].axis("off")

        axes[2, i].imshow(disp_b if disp_b.ndim == 3 else disp_b, cmap="gray" if disp_b.ndim == 2 else None)
        axes[2, i].set_title(f"Blended Band (L{i})")
        axes[2, i].axis("off")

    plt.tight_layout()
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

def make_vertical_mask(h: int, w: int, split_ratio: float = 0.5) -> np.ndarray:
    """Creates a step function binary mask: 1 on the left, 0 on the right."""
    mask = np.zeros((h, w), dtype=np.float64)
    split_col = int(w * split_ratio)
    mask[:, :split_col] = 1.0
    return mask


def make_circular_mask(h: int, w: int, center: tuple = None, radius: int = None) -> np.ndarray:
    """Creates a circular irregular mask: 1 inside the circle, 0 outside."""
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(h, w) // 4

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = (dist_from_center <= radius).astype(np.float64)
    return mask

def load_and_resize(filepath: str, max_dim: int = 1000) -> np.ndarray:
        img = load_image(filepath)
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

def load_mask(filepath: str, target_shape: tuple = None) -> np.ndarray:
    """Loads a grayscale or B&W mask image and normalizes it to float [0.0, 1.0]."""
    raw = plt.imread(filepath)
    if raw.ndim == 3:
        mask = 0.299 * raw[..., 0] + 0.587 * raw[..., 1] + 0.114 * raw[..., 2]
    else:
        mask = raw.copy()
        
    if mask.max() > 1.0:
        mask = mask / 255.0

    mask = (mask > 0.5).astype(np.float64)

    # Resize mask to match target height and width (h, w)
    if target_shape is not None:
        target_h, target_w = target_shape[:2]
        if mask.shape[:2] != (target_h, target_w):
            mask = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    return mask

def run_part2_4():
    print("\n--- Running Part 2.4: Multiresolution Blending ---")
    media_dir = "media"
    os.makedirs(media_dir, exist_ok=True)

    # Configuration list for all blends:
    blend_experiments = [
        # 1. The Oraple (Vertical Seam)
        {
            "name": "oraple",
            "im1_file": "apple.jpeg",
            "im2_file": "orange.jpeg",
            "mask_type": "vertical",  # Options: 'vertical', 'horizontal', 'circle', 'file'
            "num_levels": 5,
            "start_sigma": 2.0,
            "show_process": True,     # Generates the Figure 10 breakdown
        },
        # 2. Sunset View (Vertical Seam)
        {
            "name": "sunset",
            "im1_file": "day.jpg",          
            "im2_file": "night.jpg",         
            "mask_type": "vertical",
            "num_levels": 6,
            "start_sigma": 4.0,
            "show_process": False,
        },
        # 3. Fitting Room
        {
            "name": "clothes_tryon",
            "im1_file": "sweater.jpg",
            "im2_file": "full_body.jpg",
            "mask_type": "file",                    # Uses drawn mask file
            "mask_file": "sweater_mask.jpg",
            "num_levels": 5,
            "start_sigma": 2.0,
            "show_process": True,
        },
    ]

    for exp in blend_experiments:
        name = exp["name"]
        p1 = os.path.join(media_dir, exp["im1_file"])
        p2 = os.path.join(media_dir, exp["im2_file"])

        if not os.path.exists(p1) or not os.path.exists(p2):
            print(f"Skipping '{name}': Image files not found.")
            continue

        print(f"\nProcessing Blend: {name}...")
        im1 = load_and_resize(p1, max_dim=1000)
        im2 = load_and_resize(p2, max_dim=1000)

        # 1. Match dimensions
        h = min(im1.shape[0], im2.shape[0])
        w = min(im1.shape[1], im2.shape[1])
        im1 = im1[:h, :w]
        im2 = im2[:h, :w]

        # 2. Build the requested mask
        mtype = exp.get("mask_type", "vertical")
        if mtype == "vertical":
            mask = make_vertical_mask(h, w, split_ratio=0.5)
        elif mtype == "horizontal":
            mask = np.zeros((h, w), dtype=np.float64)
            mask[:h // 2, :] = 1.0
        elif mtype == "circle":
            r = int(min(h, w) * exp.get("circle_radius_ratio", 0.25))
            mask = make_circular_mask(h, w, radius=r)
        elif mtype == "file":
            mask_path = os.path.join(media_dir, exp["mask_file"])
            if not os.path.exists(mask_path):
                print(f"Mask file {mask_path} not found. Skipping {name}.")
                continue
            mask = load_mask(mask_path, target_shape=(h, w))
        else:
            mask = make_vertical_mask(h, w)

        # 3. Perform multiresolution blend
        blended, b_stack, l1_m, l2_m, gm = multiresolution_blend(
            im1, im2, mask,
            num_levels=exp.get("num_levels", 5),
            start_sigma=exp.get("start_sigma", 2.0)
        )

        # 4. Save results
        plt.imsave(os.path.join(media_dir, f"part2_4_{name}_blended.jpg"), blended)
        plt.imsave(os.path.join(media_dir, f"part2_4_{name}_mask.jpg"), mask, cmap="gray")

        # 5. Optionally save stack breakdown grid (Figure 10 / Figure 11)
        if exp.get("show_process", False):
            plot_blending_process(
                l1_m, l2_m, b_stack, blended,
                exp["im1_file"].split('.')[0],
                exp["im2_file"].split('.')[0],
                os.path.join(media_dir, f"part2_4_{name}_process_grid.jpg")
            )

    print("\nPart 2.4 processing finished!")

if __name__ == "__main__":
    run_part1_1()
    run_part1_2()
    run_part1_3()
    run_part1_bells()
    run_part2_1()
    run_part2_2()
    test_color_hybrids()
    run_part2_3()
    run_part2_4()

