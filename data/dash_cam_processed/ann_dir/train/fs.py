import cv2
import numpy as np
import matplotlib.pyplot as plt

label_path = '/media/yang/22667A296679FDBB/Users/Austin/Documents/MEng_Courses/Mobile-Seed-MIE1517/data/dash_cam_processed/ann_dir/train/frame_0017.png'  # the semantic label PNG
label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)

print("Unique labels in mask:", np.unique(label))