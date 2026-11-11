import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import cdist

# Import helper functions from harris.py and parta.py
from harris import get_harris_corners, dist2
from parta import computeH, warpImageBilinear, compute_mosaic_bounding_box


# ==============================================================================
# HELPER: IMAGE LOADING & DOWNSCALING (METHOD 1)
# ==============================================================================

def load_image_scaled(path, max_dim=1200):
    """
    Loads an image and caps its maximum dimension to max_dim (preserving aspect ratio)
    to prevent peak_local_max from hanging on dense, high-resolution textures.
    """
    pil_im = Image.open(path).convert("RGB")
    w, h = pil_im.size
    if max(w, h) > max_dim:
        scale = max_dim / float(max(w, h))
        new_w, new_h = int(round(w * scale)), int(round(h * scale))
        pil_im = pil_im.resize((new_w, new_h), Image.Resampling.LANCZOS)
    return np.array(pil_im, dtype=np.float64) / 255.0


# ==============================================================================
# B.1: ADAPTIVE NON-MAXIMAL SUPPRESSION (ANMS)
# ==============================================================================

def anms(coords, h_vals, num_best=500, c_robust=0.9, max_candidates=3000):
    """
    Implements Adaptive Non-Maximal Suppression (Brown et al., Section 3).
    Pre-filters to max_candidates strongest Harris peaks to avoid OOM crashes,
    and computes squared Euclidean distances in float64 to prevent overflow errors.
    
    coords: (2, N) numpy array representing (ys, xs).
    h_vals: (N,) corner strength response vector.
    """
    N = coords.shape[1]
    if N <= num_best:
        return coords

    # 1. Pre-filter to the top strongest points for fast processing
    if N > max_candidates:
        top_indices = np.argsort(-h_vals)[:max_candidates]
        coords = coords[:, top_indices]
        h_vals = h_vals[top_indices]
        N = max_candidates

    # Ensure coordinates are float64 so dist2 supports np.inf assignment
    pts = np.vstack([coords[1], coords[0]]).T.astype(np.float64)  # (N, 2): [x, y]

    # 2. Vectorized pairwise squared Euclidean distance (diff is N x N x 2)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_matrix = np.sum(diff ** 2, axis=-1)

    # 3. Suppress neighbors that are not significantly stronger
    stronger_mask = h_vals[:, None] < (c_robust * h_vals[None, :])
    dist_matrix[~stronger_mask] = np.inf

    # Minimum radius to a significantly stronger corner
    r2 = np.min(dist_matrix, axis=1)

    # Pick the top num_best points with the largest suppression radii
    best_indices = np.argsort(-r2)[:num_best]
    return coords[:, best_indices]


# ==============================================================================
# B.2: FEATURE DESCRIPTOR EXTRACTION
# ==============================================================================

def extract_feature_descriptors(im_gray, coords, patch_window=40, desc_dim=8):
    """
    Samples a large (patch_window x patch_window) window around each feature point,
    applies Gaussian anti-aliasing blur, downsamples to (desc_dim x desc_dim),
    and normalizes mean to 0 and standard deviation to 1.
    
    coords: (2, K) array of (ys, xs).
    Returns:
        descriptors: (M, desc_dim * desc_dim) normalized descriptor vectors.
        valid_coords: (2, M) corresponding (ys, xs) points that fit within boundaries.
    """
    half_w = patch_window // 2
    step = patch_window // desc_dim
    H_img, W_img = im_gray.shape

    # Pre-blur the grayscale image to prevent aliasing when downsampling
    blurred = gaussian_filter(im_gray, sigma=1.0)

    descriptors = []
    valid_indices = []

    for idx in range(coords.shape[1]):
        y, x = int(coords[0, idx]), int(coords[1, idx])

        # Boundary check for full patch window
        if (y - half_w < 0 or y + half_w > H_img or 
            x - half_w < 0 or x + half_w > W_img):
            continue

        # Extract patch_window x patch_window window and sample every `step` pixels -> 8x8
        patch = blurred[y - half_w : y + half_w, x - half_w : x + half_w]
        desc = patch[::step, ::step].astype(np.float64)

        # Bias and gain normalization (zero mean, unit std)
        mean = np.mean(desc)
        std = np.std(desc)
        if std < 1e-5:
            desc = desc - mean  # Flat intensity patch
        else:
            desc = (desc - mean) / std

        descriptors.append(desc.flatten())
        valid_indices.append(idx)

    descriptors = np.array(descriptors, dtype=np.float64)
    valid_coords = coords[:, valid_indices]
    return descriptors, valid_coords


# ==============================================================================
# B.3: FEATURE MATCHING (LOWE'S RATIO TEST)
# ==============================================================================

def match_features(desc1, desc2, pts1, pts2, ratio_thresh=0.65):
    """
    Finds matching feature pairs using Lowe's 1st-to-2nd nearest neighbor ratio test.
    pts1, pts2 are (2, M) arrays representing (ys, xs).
    Returns:
        matched_pts1: (K, 2) array of (x, y) coordinates in Image 1.
        matched_pts2: (K, 2) array of (x, y) coordinates in Image 2.
    """
    if len(desc1) == 0 or len(desc2) == 0:
        return np.empty((0, 2)), np.empty((0, 2))

    # Compute Euclidean distance matrix between all descriptor pairs
    dists = cdist(desc1, desc2, metric='euclidean')  # (N1, N2)

    matched_pts1 = []
    matched_pts2 = []

    for i in range(dists.shape[0]):
        # Find 1st and 2nd nearest neighbors in Image 2
        sorted_indices = np.argsort(dists[i])
        idx1, idx2 = sorted_indices[0], sorted_indices[1]
        e1, e2 = dists[i, idx1], dists[i, idx2]

        # Lowe's trick: filter out ambiguous matches
        if e1 / (e2 + 1e-12) < ratio_thresh:
            matched_pts1.append([pts1[1, i], pts1[0, i]])       # store as (x, y)
            matched_pts2.append([pts2[1, idx1], pts2[0, idx1]])  # store as (x, y)

    return np.array(matched_pts1, dtype=np.float64), np.array(matched_pts2, dtype=np.float64)


# ==============================================================================
# B.4: 4-POINT RANSAC FOR ROBUST HOMOGRAPHY
# ==============================================================================

def ransac_homography(pts1, pts2, num_iters=3000, inlier_thresh=3.0):
    """
    4-point RANSAC to compute a robust homography mapping pts1 -> pts2.
    pts1, pts2: (N, 2) arrays of (x, y) matching coordinates.
    Returns:
        best_H: 3x3 homography matrix recomputed on all inliers.
        inlier_mask: boolean mask indicating inlier point correspondences.
    """
    N = pts1.shape[0]
    if N < 4:
        raise ValueError("At least 4 correspondence pairs required for RANSAC.")

    best_inliers = []
    pts1_homo = np.hstack([pts1, np.ones((N, 1))]).T  # (3, N)

    for _ in range(num_iters):
        # 1. Randomly sample 4 points
        sample_indices = np.random.choice(N, 4, replace=False)
        sample1 = pts1[sample_indices]
        sample2 = pts2[sample_indices]

        try:
            H_cand = computeH(sample1, sample2)
        except (np.linalg.LinAlgError, ValueError):
            continue

        # 2. Project all points from Image 1: p2_est ~ H * p1
        proj = H_cand @ pts1_homo
        proj /= (proj[2, :] + 1e-12)
        proj_pts2 = proj[:2, :].T  # (N, 2)

        # 3. Compute Euclidean reprojection error ||p2 - p2_est||
        errors = np.linalg.norm(pts2 - proj_pts2, axis=1)
        inliers = np.where(errors < inlier_thresh)[0]

        if len(inliers) > len(best_inliers):
            best_inliers = inliers

    if len(best_inliers) < 4:
        raise RuntimeError("RANSAC failed to find at least 4 consistent inliers.")

    # 4. Refine final H using all consensus inliers via least-squares
    refined_H = computeH(pts1[best_inliers], pts2[best_inliers])
    inlier_mask = np.zeros(N, dtype=bool)
    inlier_mask[best_inliers] = True

    return refined_H, inlier_mask


# ==============================================================================
# VISUALIZATION DELIVERABLES EXPORTERS
# ==============================================================================

def save_corners_overlay(im_rgb, raw_coords, anms_coords, out_path):
    """Visualizes raw Harris points vs ANMS filtered points side-by-side."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    axes[0].imshow(im_rgb)
    axes[0].scatter(raw_coords[1], raw_coords[0], s=3, c='red', alpha=0.5)
    axes[0].set_title(f"Raw Harris Corners ({raw_coords.shape[1]} points)")
    axes[0].axis("off")

    axes[1].imshow(im_rgb)
    axes[1].scatter(anms_coords[1], anms_coords[0], s=14, c='cyan')
    axes[1].set_title(f"Adaptive Non-Maximal Suppression ({anms_coords.shape[1]} points)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_sample_descriptors(descriptors, out_path, num_samples=8):
    """Visualizes several extracted 8x8 normalized descriptor patches."""
    fig, axes = plt.subplots(1, num_samples, figsize=(14, 2.5))
    for i in range(min(num_samples, len(descriptors))):
        patch = descriptors[i].reshape(8, 8)
        axes[i].imshow(patch, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f"#{i+1}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


def save_matches_visualization(im1_rgb, im2_rgb, pts1, pts2, inlier_mask, out_path):
    """Plots side-by-side matches with connecting lines (inliers vs outliers)."""
    h1, w1 = im1_rgb.shape[:2]
    h2, w2 = im2_rgb.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=im1_rgb.dtype)
    canvas[:h1, :w1] = im1_rgb
    canvas[:h2, w1:w1+w2] = im2_rgb

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.imshow(canvas)
    
    # Plot connecting lines
    for i in range(len(pts1)):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i][0] + w1, pts2[i][1]
        if inlier_mask[i]:
            ax.plot([x1, x2], [y1, y2], color='lime', linewidth=1.0, alpha=0.8)
            ax.scatter([x1, x2], [y1, y2], color='yellow', s=10)
        else:
            ax.plot([x1, x2], [y1, y2], color='red', linewidth=0.5, linestyle='--', alpha=0.3)

    ax.set_title(f"Feature Matches: {np.sum(inlier_mask)} Inliers (Green) / {len(pts1) - np.sum(inlier_mask)} Outliers (Red)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# ==============================================================================
# PIPELINE EXECUTION FOR AUTOSTITCHING
# ==============================================================================

def run_autostitch(im1_path, im2_path, pair_id="1"):
    print(f"\n=======================================================")
    print(f" Processing Autostitch: Mosaic {pair_id}")
    print(f"=======================================================")
    
    # Load and scale to max 1200px to avoid freeze on dense textures
    im1 = load_image_scaled(im1_path, max_dim=1200)
    im2 = load_image_scaled(im2_path, max_dim=1200)

    im1_gray = rgb2gray(im1)
    im2_gray = rgb2gray(im2)

    # 1. Harris Corner Detection
    print("1. Detecting Harris Corners...")
    h1, coords1 = get_harris_corners(im1_gray, edge_discard=20)
    h2, coords2 = get_harris_corners(im2_gray, edge_discard=20)

    # Extract corner response intensities at detected peaks
    vals1 = h1[coords1[0], coords1[1]]
    vals2 = h2[coords2[0], coords2[1]]

    # 2. Adaptive Non-Maximal Suppression (ANMS)
    print("2. Running ANMS...")
    anms_coords1 = anms(coords1, vals1, num_best=500, c_robust=0.9, max_candidates=3000)
    anms_coords2 = anms(coords2, vals2, num_best=500, c_robust=0.9, max_candidates=3000)

    # Save ANMS deliverable for webpage
    save_corners_overlay(im1, coords1, anms_coords1, f"output/mosaic_{pair_id}_anms_corners.jpg")

    # 3. Extract 8x8 Feature Descriptors
    print("3. Extracting 8x8 normalized descriptors...")
    desc1, valid_pts1 = extract_feature_descriptors(im1_gray, anms_coords1, patch_window=40, desc_dim=8)
    desc2, valid_pts2 = extract_feature_descriptors(im2_gray, anms_coords2, patch_window=40, desc_dim=8)

    save_sample_descriptors(desc1, f"output/mosaic_{pair_id}_sample_descriptors.jpg")

    # 4. Feature Matching via Lowe's Ratio Test
    print("4. Matching features with Lowe's ratio test...")
    matched_pts1, matched_pts2 = match_features(desc1, desc2, valid_pts1, valid_pts2, ratio_thresh=0.65)
    print(f"   Found {len(matched_pts1)} candidate matches.")

    # 5. Robust Homography Recovery via 4-Point RANSAC (maps im2 -> im1)
    print("5. Computing RANSAC homography...")
    H_2to1, inlier_mask = ransac_homography(matched_pts2, matched_pts1, num_iters=3000, inlier_thresh=3.0)
    print(f"   RANSAC consensus: {np.sum(inlier_mask)} / {len(matched_pts1)} inliers.")

    # Save match lines visualization
    save_matches_visualization(im1, im2, matched_pts1, matched_pts2, inlier_mask, f"output/mosaic_{pair_id}_ransac_matches.jpg")

    # 6. Stitch and Blend Panorama (reusing Part A distance-transform blending)
    print("6. Warping and blending final automatic mosaic...")
    out_shape, offset = compute_mosaic_bounding_box(im1.shape, im2.shape, H_2to1)
    H_out, W_out = out_shape
    ox, oy = offset

    canvas1 = np.zeros((H_out, W_out, 3), dtype=np.float64)
    mask1 = np.zeros((H_out, W_out), dtype=bool)
    y0, y1 = -oy, -oy + im1.shape[0]
    x0, x1 = -ox, -ox + im1.shape[1]
    canvas1[y0:y1, x0:x1] = im1
    mask1[y0:y1, x0:x1] = True

    canvas2, mask2 = warpImageBilinear(im2, H_2to1, out_shape=out_shape, offset=offset)

    from scipy.ndimage import distance_transform_edt
    dist1 = distance_transform_edt(mask1)
    dist2 = distance_transform_edt(mask2)
    weight_sum = dist1 + dist2
    weight_sum[weight_sum == 0] = 1.0

    w1 = (dist1 / weight_sum)[:, :, None]
    w2 = (dist2 / weight_sum)[:, :, None]
    blended = np.clip(canvas1 * w1 + canvas2 * w2, 0.0, 1.0)

    out_file = f"output/mosaic_{pair_id}_autostitched.jpg"
    Image.fromarray((blended * 255).astype(np.uint8)).save(out_file)
    print(f"   Finished! Saved automatic panorama to {out_file}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Automatically process all 3 mosaic sets from Part A
    run_autostitch("media/part_a/mosaic_1a.jpg", "media/part_a/mosaic_1b.jpg", pair_id="1")
    run_autostitch("media/part_a/mosaic_2a.jpeg", "media/part_a/mosaic_2b.jpeg", pair_id="2")
    run_autostitch("media/part_a/mosaic_3a.jpg", "media/part_a/mosaic_3b.jpg", pair_id="3")