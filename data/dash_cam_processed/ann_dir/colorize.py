import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load grayscale mask
img_path = "/media/yang/22667A296679FDBB/Users/Austin/Documents/MEng_Courses/Mobile-Seed-MIE1517/data/dash_cam_processed/ann_dir/train/frame_0004.png"
mask = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

# Cityscapes-style class index to color mapping (subset)
color_palette = {
    0: [128, 64, 128],    # road
    1: [244, 35, 232],    # sidewalk
    2: [70, 70, 70],      # building
    3: [102, 102, 156],   # wall
    4: [190, 153, 153],   # fence
    5: [153, 153, 153],   # pole
    6: [250, 170, 30],    # traffic light
    7: [220, 220, 0],     # traffic sign
    8: [107, 142, 35],    # vegetation
    10: [70, 130, 180],   # sky
    11: [220, 20, 60],    # person
    12: [255, 0, 0],      # rider
    13: [0, 0, 142],      # car
    14: [0, 0, 70],       # truck
    15: [0, 60, 100],     # bus
    18: [0, 0, 230],      # motorcycle
    255: [0, 0, 0],       # ignored
}

# Create a color version of the mask
color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
for label, color in color_palette.items():
    color_mask[mask == label] = color

# Convert to RGB for plotting
color_mask_rgb = cv2.cvtColor(color_mask, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(12, 6))
plt.imshow(color_mask_rgb)
plt.title("Colorized Semantic Mask")
plt.axis("off")
plt.show()
