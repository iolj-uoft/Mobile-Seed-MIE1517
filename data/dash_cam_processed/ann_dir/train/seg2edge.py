import numpy as np
import cv2
import os
from glob import glob
from tqdm import tqdm

def seg2edge(mask, radius=2):
    """
    Converts a segmentation mask to an edge map.
    Args:
        mask: 2D numpy array of label IDs.
        radius: Neighborhood radius for edge detection.
    Returns:
        edge: Binary edge map (0 or 255).
    """
    h, w = mask.shape
    edge = np.zeros((h, w), dtype=np.uint8)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dx == 0 and dy == 0:
                continue
            shifted = np.roll(np.roll(mask, dy, axis=0), dx, axis=1)
            edge |= (shifted != mask).astype(np.uint8)

    edge = (edge > 0).astype(np.uint8) * 255
    return edge

# === Batch Processing Example ===
def process_folder(input_folder, output_folder, radius=2):
    os.makedirs(output_folder, exist_ok=True)
    mask_paths = sorted(glob(os.path.join(input_folder, '*.png')))
    
    for mask_path in tqdm(mask_paths):
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        edge = seg2edge(mask, radius=radius)
        filename = os.path.basename(mask_path).replace('.png', '_edge.png')
        out_path = os.path.join(output_folder, filename)
        cv2.imwrite(out_path, edge)

# === Example usage ===
process_folder("data/dash_cam_processed/ann_dir/train", "data/dash_cam_processed/edge_dir/train", radius=2)
