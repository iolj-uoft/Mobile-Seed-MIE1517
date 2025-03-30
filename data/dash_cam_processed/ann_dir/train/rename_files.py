import os
import glob
import shutil

input_dir = 'data/dash_cam_processed/ann_dir/train'
output_dir = 'data/dash_cam_processed/ann_dir/train'
os.makedirs(output_dir, exist_ok=True)

for path in glob.glob(os.path.join(input_dir, '*_labelIds.png')):
    filename = os.path.basename(path)
    new_name = filename.replace('_gtFine_labelIds', '').replace('processed_', '')
    dst = os.path.join(output_dir, new_name)
    shutil.copy(path, dst)  # or shutil.move(path, dst) to move instead
    print(f'✅ Renamed to: {new_name}')