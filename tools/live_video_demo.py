import os
import cv2
import torch
from mmseg.apis import init_segmentor, inference_segmentor
import numpy as np

# --- Get user input for prefix ---
prefix = input("Enter a prefix for output filename (e.g., 'seg'): ").strip()

# --- CONFIGS ---
config_file = 'configs/Mobile_Seed/MS_tiny_cityscapes.py'
checkpoint_file = 'ckpt/GCA.pth'
device = 'cuda:0'
video_path = 'data/dash_cam/Stockyards.ts'  # Replace with your .ts file

# --- Construct output filename ---
video_name = os.path.splitext(os.path.basename(video_path))[0]
output_path = f'demo/{video_name}_{prefix}.mp4'

# Cityscapes palette in RGB
CITYSCAPES_PALETTE = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32]
]
palette_bgr = [list(reversed(c)) for c in CITYSCAPES_PALETTE]

# --- Load model ---
model = init_segmentor(config_file, checkpoint_file, device=device)

# --- Open input video ---
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = 1280, 720  # Output resolution

# --- Create output video writer ---
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# --- Process ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.resize(frame_rgb, (1024, 512))

    # Inference
    with torch.no_grad():
        result = inference_segmentor(model, frame_rgb)

    # Visualize
    overlay = model.show_result(
        frame_rgb,
        result,
        show=False,
        opacity=0.5,
        palette=palette_bgr
    )

    # Resize for output
    output_frame = cv2.resize(overlay, (width, height))
    output_bgr = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

    # Write frame
    out.write(output_bgr)

    # (Optional) Show live
    cv2.imshow('Segmented Video', output_bgr)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Saved output video to: {output_path}")
