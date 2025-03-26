import json
import os
import cv2
from labelme.utils import img_b64_to_arr
import numpy as np
import matplotlib.pyplot as plt

import json
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from labelme.utils import img_b64_to_arr

def visualize_labelme_annotation(json_path, show=True, save_path=None):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Decode image using labelme's method
    if 'imageData' in data:
        image = img_b64_to_arr(data['imageData'])
    else:
        # Load image from file path
        img_dir = os.path.dirname(json_path)
        img_path = os.path.join(img_dir, data['imagePath'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image at: {img_path}")

    overlay = image.copy()
    for shape in data['shapes']:
        points = np.array(shape['points'], dtype=np.int32)
        label = shape['label']
        cv2.polylines(overlay, [points], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(overlay, label, tuple(points[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)

    if save_path:
        cv2.imwrite(save_path, blended)
        print(f"Saved visualized overlay to {save_path}")

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title(os.path.basename(json_path))
        plt.show()


# Example usage:
if __name__ == "__main__":
    import sys
    json_file = sys.argv[1] if len(sys.argv) > 1 else r'C:\Users\Austin\Documents\MEng_Courses\Mobile-Seed-MIE1517\finetuning\outseg_pedestrians.json'
    visualize_labelme_annotation(json_file)
