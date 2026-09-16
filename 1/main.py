import numpy as np
import skimage as sk
import skimage.io as skio
import skimage.transform as sktr
from skimage.filters import sobel

# 1. Image Alignment (NCC Metric)
def align_single_scale(ref, target, search_window=15):
    best_score = -np.inf
    best_shift = (0, 0)

    # Standard 10% interior cropping margin
    h, w = ref.shape
    crop_h, crop_w = int(0.10 * h), int(0.10 * w)

    # Fixed reference patch
    ref_crop = ref[crop_h : h - crop_h, crop_w : w - crop_w]
    ref_zero = ref_crop - np.mean(ref_crop)
    ref_norm = ref_zero / (np.linalg.norm(ref_zero) + 1e-8)

    for dy in range(-search_window, search_window + 1):
        for dx in range(-search_window, search_window + 1):
            # Shift target directly with np.roll
            shifted_target = np.roll(target, shift=(dy, dx), axis=(0, 1))
            target_crop = shifted_target[crop_h : h - crop_h, crop_w : w - crop_w]

            target_zero = target_crop - np.mean(target_crop)
            target_norm = target_zero / (np.linalg.norm(target_zero) + 1e-8)

            score = np.sum(ref_norm * target_norm)
            if score > best_score:
                best_score = score
                best_shift = (dy, dx)

    return best_shift

def align_pyramid(ref, target, min_size=250):
    h, w = ref.shape

    # Base case: image is coarse enough for exhaustive search
    if h <= min_size or w <= min_size:
        return align_single_scale(ref, target, search_window=15)

    # Recursive step: shrink by half (anti_aliasing=False is fast)
    ref_down = sktr.rescale(ref, 0.5, anti_aliasing=False)
    target_down = sktr.rescale(target, 0.5, anti_aliasing=False)

    # Recurse
    coarse_dy, coarse_dx = align_pyramid(ref_down, target_down, min_size=min_size)

    # Scale displacement estimate up by 2
    guess_y = coarse_dy * 2
    guess_x = coarse_dx * 2

    # Pre-shift target by coarse guess
    target_pre_shifted = np.roll(target, shift=(guess_y, guess_x), axis=(0, 1))

    # Fine search [-2, 2] around that guess
    refine_y, refine_x = align_single_scale(ref, target_pre_shifted, search_window=2)

    return (guess_y + refine_y, guess_x + refine_x)

# 2. Image Processing Pipeline
def process_image(image_name, use_edges=False):
    input_path = f"1/media/{image_name}"
    try:
        im = skio.imread(input_path)
    except FileNotFoundError:
        print(f"Skipping {image_name}: File not found.")
        return 

    mode = "Edges (Sobel)" if use_edges else "Raw NCC"
    print(f"\nProcessing: {image_name} [{mode}]")
    im = sk.img_as_float(im)

    # Split channels (BGR from top to bottom)
    height = im.shape[0] // 3
    b = im[0:height, :]
    g = im[height : 2 * height, :]
    r = im[2 * height : 3 * height, :]

    # Choose features for alignment
    if use_edges:
        ref_b, target_g, target_r = sobel(b), sobel(g), sobel(r)
    else:
        ref_b, target_g, target_r = b, g, r

    # Compute shifts
    g_shift = align_pyramid(ref_b, target_g)
    r_shift = align_pyramid(ref_b, target_r)

    print(f"Green shift (dy, dx): {g_shift}")
    print(f"Red shift (dy, dx): {r_shift}")

    # Always shift the ORIGINAL color channels
    g_aligned = np.roll(g, shift=g_shift, axis=(0, 1))
    r_aligned = np.roll(r, shift=r_shift, axis=(0, 1))

    color_im = np.dstack((r_aligned, g_aligned, b))
    color_im = np.clip(color_im, 0.0, 1.0)

    # Add a suffix if using edges so it doesn't overwrite the raw version
    base_name = image_name.rsplit(".", 1)[0]
    suffix = "_aligned_edge.jpg" if use_edges else "_aligned.jpg"
    out_fname = f"1/media/{base_name}{suffix}"

    skio.imsave(out_fname, sk.img_as_ubyte(color_im))
    print(f"Saved: {out_fname}")

if __name__ == "__main__":
    # List all test images
    test_images = [
        # Small JPEGs
        "cathedral.jpg", "monastery.jpg", "tobolsk.jpg", "vodopad.jpg",
        # Large TIFFs
        "church.tif", "emir.tif", "harvesters.tif", "icon.tif", "ilemselga.tif", "melons.tif", "religious_painting.tif", 
        "self_portrait.tif", "siren.tif", "three_generations.tif", "wharf.tif", "laika.tif", "rabota.tif"
        ]

    # Standard Raw NCC
    for img in test_images:
        process_image(img, use_edges=False)

    # Bells & Whistles run for Emir of Bukhara
    process_image("emir.tif", use_edges=True)
