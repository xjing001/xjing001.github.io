import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import distance_transform_edt

# A.1 & A.2: Point Selection & Visualization

def get_or_load_correspondences(im1, im2, cache_path):
    """
    Interactively selects or loads cached point pairs.
    Saves visual correspondence verification plots.
    """
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
            im1_pts = np.array(data["im1_pts"], dtype=np.float64)
            im2_pts = np.array(data["im2_pts"], dtype=np.float64)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        axes[0].imshow(im1)
        axes[0].set_title("Click point on Image 1")
        axes[1].imshow(im2)
        axes[1].set_title("Click corresponding point on Image 2")
        im1_pts, im2_pts = [], []

        print(f"Selecting points for {cache_path}. Alternate: Img1 -> Img2.")
        print("Press Enter or middle-click in the plot window when done (>= 4 pairs).")
        while True:
            pt1 = plt.ginput(1, timeout=-1)
            if not pt1:
                break
            im1_pts.append(pt1[0])
            axes[0].plot(pt1[0][0], pt1[0][1], "ro")
            fig.canvas.draw()

            pt2 = plt.ginput(1, timeout=-1)
            if not pt2:
                im1_pts.pop()
                break
            im2_pts.append(pt2[0])
            axes[1].plot(pt2[0][0], pt2[0][1], "go")
            fig.canvas.draw()

        plt.close(fig)
        im1_pts = np.array(im1_pts, dtype=np.float64)
        im2_pts = np.array(im2_pts, dtype=np.float64)

        with open(cache_path, "w") as f:
            json.dump({"im1_pts": im1_pts.tolist(), "im2_pts": im2_pts.tolist()}, f, indent=2)

    return im1_pts, im2_pts


def save_correspondence_visualization(im1, im2, pts1, pts2, out_path):
    """Visualizes clicked correspondences side-by-side with indexed tags."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    axes[0].imshow(im1)
    axes[0].scatter(pts1[:, 0], pts1[:, 1], c='red', s=30)
    for i, pt in enumerate(pts1):
        axes[0].annotate(str(i), (pt[0], pt[1]), color='yellow', fontsize=11, weight='bold')
    axes[0].set_title("Image 1 Correspondences")
    axes[0].axis("off")

    axes[1].imshow(im2)
    axes[1].scatter(pts2[:, 0], pts2[:, 1], c='lime', s=30)
    for i, pt in enumerate(pts2):
        axes[1].annotate(str(i), (pt[0], pt[1]), color='yellow', fontsize=11, weight='bold')
    axes[1].set_title("Image 2 Correspondences")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=200)
    plt.close(fig)


# A.2: Recover Homographies (Least-Squares Formulation)

def computeH(im1_pts, im2_pts):
    """
    Computes 3x3 homography H mapping p1 -> p2 (p2 ~ H * p1).
    Sets up Ah = b with H[2, 2] = 1 and solves using np.linalg.lstsq.
    """
    n = im1_pts.shape[0]
    if n < 4:
        raise ValueError("At least 4 correspondences are required to compute H.")

    A = []
    b = []
    for i in range(n):
        x, y = im1_pts[i]
        xp, yp = im2_pts[i]

        A.append([x, y, 1.0, 0.0, 0.0, 0.0, -x * xp, -y * xp])
        A.append([0.0, 0.0, 0.0, x, y, 1.0, -x * yp, -y * yp])
        b.extend([xp, yp])

    A = np.asarray(A, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    h, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    H = np.append(h, 1.0).reshape(3, 3)
    return H


# A.3: Inverse Warping & Custom Interpolations

def warpImageNearestNeighbor(im, H, out_shape, offset=(0, 0)):
    """Nearest Neighbor interpolation"""
    H_inv = np.linalg.inv(H)
    H_in, W_in = im.shape[:2]
    H_out, W_out = out_shape
    ox, oy = offset

    grid_x, grid_y = np.meshgrid(np.arange(W_out) + ox, np.arange(H_out) + oy)
    pts_out = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)], axis=0)

    pts_src = H_inv @ pts_out
    pts_src /= (pts_src[2, :] + 1e-12)

    src_x = np.round(pts_src[0, :]).astype(int)
    src_y = np.round(pts_src[1, :]).astype(int)

    valid_mask = ((src_x >= 0) & (src_x < W_in) & (src_y >= 0) & (src_y < H_in)).reshape(H_out, W_out)
    src_x_valid = np.clip(src_x, 0, W_in - 1).reshape(H_out, W_out)
    src_y_valid = np.clip(src_y, 0, H_in - 1).reshape(H_out, W_out)

    warped = np.zeros((H_out, W_out, 3), dtype=np.float64)
    for c in range(3):
        warped[:, :, c] = im[src_y_valid, src_x_valid, c] * valid_mask

    return warped, valid_mask


def warpImageBilinear(im, H, out_shape, offset=(0, 0)):
    """Bilinear interpolation"""
    H_inv = np.linalg.inv(H)
    H_in, W_in = im.shape[:2]
    H_out, W_out = out_shape
    ox, oy = offset

    grid_x, grid_y = np.meshgrid(np.arange(W_out) + ox, np.arange(H_out) + oy)
    pts_out = np.stack([grid_x.ravel(), grid_y.ravel(), np.ones(grid_x.size)], axis=0)

    pts_src = H_inv @ pts_out
    pts_src /= (pts_src[2, :] + 1e-12)

    x_s = pts_src[0, :]
    y_s = pts_src[1, :]

    x0 = np.floor(x_s).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y_s).astype(int)
    y1 = y0 + 1

    valid_mask = ((x0 >= 0) & (x1 < W_in) & (y0 >= 0) & (y1 < H_in)).reshape(H_out, W_out)

    dx = (x_s - x0).reshape(H_out, W_out)
    dy = (y_s - y0).reshape(H_out, W_out)

    x0_cl = np.clip(x0, 0, W_in - 1).reshape(H_out, W_out)
    x1_cl = np.clip(x1, 0, W_in - 1).reshape(H_out, W_out)
    y0_cl = np.clip(y0, 0, H_in - 1).reshape(H_out, W_out)
    y1_cl = np.clip(y1, 0, H_in - 1).reshape(H_out, W_out)

    w00 = (1.0 - dx) * (1.0 - dy)
    w10 = dx * (1.0 - dy)
    w01 = (1.0 - dx) * dy
    w11 = dx * dy

    warped = np.zeros((H_out, W_out, 3), dtype=np.float64)
    for c in range(3):
        val = (w00 * im[y0_cl, x0_cl, c] +
               w10 * im[y0_cl, x1_cl, c] +
               w01 * im[y1_cl, x0_cl, c] +
               w11 * im[y1_cl, x1_cl, c])
        warped[:, :, c] = val * valid_mask

    return warped, valid_mask


# A.4: Bounding Box & Distance-Transform Feathering

def compute_mosaic_bounding_box(im1_shape, im2_shape, H_2to1):
    """Calculates unified mosaic canvas dimensions and negative origin offsets."""
    h1, w1 = im1_shape[:2]
    h2, w2 = im2_shape[:2]

    # Map the 4 corners of im2 into im1 coordinate system
    corners2 = np.array([
        [0, 0, 1.0],
        [w2, 0, 1.0],
        [w2, h2, 1.0],
        [0, h2, 1.0]
    ]).T

    warped_c2 = H_2to1 @ corners2
    warped_c2 /= warped_c2[2, :]

    all_x = np.concatenate([[0, w1], warped_c2[0, :]])
    all_y = np.concatenate([[0, h1], warped_c2[1, :]])

    min_x, max_x = int(np.floor(np.min(all_x))), int(np.ceil(np.max(all_x)))
    min_y, max_y = int(np.floor(np.min(all_y))), int(np.ceil(np.max(all_y)))

    out_shape = (max_y - min_y, max_x - min_x)
    offset = (min_x, min_y)
    return out_shape, offset


def blend_mosaic_feathered(im1, im2, im1_pts, im2_pts):
    """
    Warp im2 to im1 reference frame and blend using distance-transform alpha weights
    to prevent visible seams.
    """
    H_2to1 = computeH(im2_pts, im1_pts)
    out_shape, offset = compute_mosaic_bounding_box(im1.shape, im2.shape, H_2to1)
    H_out, W_out = out_shape
    ox, oy = offset

    # Reference Image
    canvas1 = np.zeros((H_out, W_out, 3), dtype=np.float64)
    mask1 = np.zeros((H_out, W_out), dtype=bool)
    y0, y1 = -oy, -oy + im1.shape[0]
    x0, x1 = -ox, -ox + im1.shape[1]
    canvas1[y0:y1, x0:x1] = im1
    mask1[y0:y1, x0:x1] = True

    # Warped Image via Bilinear
    canvas2, mask2 = warpImageBilinear(im2, H_2to1, out_shape=out_shape, offset=offset)

    # Compute Euclidean distance to image boundaries for smooth alpha feathering
    dist1 = distance_transform_edt(mask1)
    dist2 = distance_transform_edt(mask2)
    weight_sum = dist1 + dist2
    weight_sum[weight_sum == 0] = 1.0

    w1 = (dist1 / weight_sum)[:, :, None]
    w2 = (dist2 / weight_sum)[:, :, None]

    blended = canvas1 * w1 + canvas2 * w2
    return np.clip(blended, 0.0, 1.0)


# Execution Script

def run_rectification(image_name, cache_name, target_size=(400, 300)):
    print(f"\n--- Processing Rectification: {image_name} ---")
    im = np.array(Image.open(image_name).convert("RGB")) / 255.0
    tw, th = target_size

    dst_corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], dtype=np.float64)

    if not os.path.exists(cache_name):
        print(f"Click the 4 corners of the rectangular object in order: TL -> TR -> BR -> BL")
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.imshow(im)
        pts = plt.ginput(4, timeout=-1)
        plt.close(fig)
        src_corners = np.array(pts, dtype=np.float64)
        with open(cache_name, "w") as f:
            json.dump({"src_corners": src_corners.tolist()}, f, indent=2)
    else:
        with open(cache_name, "r") as f:
            src_corners = np.array(json.load(f)["src_corners"], dtype=np.float64)

    H = computeH(src_corners, dst_corners)

    # Benchmark NN
    t0 = time.time()
    rect_nn, _ = warpImageNearestNeighbor(im, H, out_shape=(th, tw))
    t_nn = time.time() - t0

    # Benchmark Bilinear
    t0 = time.time()
    rect_bil, _ = warpImageBilinear(im, H, out_shape=(th, tw))
    t_bil = time.time() - t0

    print(f"Timing comparison - NN: {t_nn:.4f}s | Bilinear: {t_bil:.4f}s")

    base = os.path.splitext(os.path.basename(image_name))[0]
    Image.fromarray((rect_nn * 255).astype(np.uint8)).save(f"output/{base}_rect_nn.jpg")
    Image.fromarray((rect_bil * 255).astype(np.uint8)).save(f"output/{base}_rect_bilinear.jpg")


def run_mosaic(im1_path, im2_path, pair_id):
    """Runs correspondence recovery, alignment and distance-weighted blending."""
    print(f"\n--- Processing Mosaic {pair_id}: {im1_path} + {im2_path} ---")
    im1 = np.array(Image.open(im1_path).convert("RGB")) / 255.0
    im2 = np.array(Image.open(im2_path).convert("RGB")) / 255.0

    cache_file = f"output/pts_mosaic_{pair_id}.json"
    pts1, pts2 = get_or_load_correspondences(im1, im2, cache_file)

    save_correspondence_visualization(
        im1, im2, pts1, pts2, f"output/mosaic_{pair_id}_correspondences.jpg"
    )

    mosaic = blend_mosaic_feathered(im1, im2, pts1, pts2)
    out_file = f"output/mosaic_{pair_id}_blended.jpg"
    Image.fromarray((mosaic * 255).astype(np.uint8)).save(out_file)
    print(f"Saved mosaic result to {out_file}")


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # Run Rectifications (A.3)
    run_rectification("media/part_a/rect_1.jpg", "output/rect_1_pts.json", target_size=(400, 300))
    run_rectification("media/part_a/rect_2.jpg", "output/rect_2_pts.json", target_size=(400, 300))
    run_rectification("media/part_a/rect_3.jpeg", "output/rect_3_pts.json", target_size=(400, 300))

    # Run Mosaics (A.4)
    run_mosaic("media/part_a/mosaic_1a.jpg", "media/part_a/mosaic_1b.jpg", pair_id="1")
    run_mosaic("media/part_a/mosaic_2a.jpeg", "media/part_a/mosaic_2b.jpeg", pair_id="2")
    run_mosaic("media/part_a/mosaic_3a.jpg", "media/part_a/mosaic_3b.jpg", pair_id="3")