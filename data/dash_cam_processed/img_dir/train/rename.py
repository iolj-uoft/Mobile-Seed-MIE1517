import os

img_dir = 'data/dash_cam_processed/img_dir/train'
ann_dir = 'data/dash_cam_processed/ann_dir/train'

img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
ann_files = sorted([f for f in os.listdir(ann_dir) if f.endswith('.png')])

print(f"Found {len(img_files)} images")
print(f"Found {len(ann_files)} masks")

# Check for mismatches
mismatch = [f for f in img_files if f.replace('.jpg', '.png') not in ann_files]
if mismatch:
    print("Missing masks for:", mismatch)
