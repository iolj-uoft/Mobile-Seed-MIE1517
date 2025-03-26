import os
import json
import cv2
import numpy as np
from skimage import measure
from tqdm import tqdm

def binary_mask_to_polygons(mask, epsilon=2.0):
    contours = measure.find_contours(mask, 0.5)
    polygons = []

    for contour in contours:
        if len(contour) < 3:
            continue
        contour = np.flip(contour, axis=1)
        approx = cv2.approxPolyDP(contour.astype(np.float32), epsilon, True)
        polygon = approx[:, 0, :].astype(float).tolist()
        polygons.append(polygon)
    return polygons

def combine_masks_to_labelme(masks_folder, image_path, output_json_path, label_map):
    image = cv2.imread(image_path)
    h, w = image.shape[:2]
    shapes = []

    for filename in tqdm(sorted(os.listdir(masks_folder))):
        if filename.endswith('.png'):
            class_id = int(filename.split('_')[1].split('.')[0])
            mask_path = os.path.join(masks_folder, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

            polygons = binary_mask_to_polygons(mask)

            for poly in polygons:
                shape = {
                    "label": label_map.get(class_id, f"class_{class_id}"),
                    "points": poly,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                shapes.append(shape)

    annotation = {
        "version": "5.0.1",
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w
    }

    with open(output_json_path, 'w') as f:
        json.dump(annotation, f, indent=4)

    return f"Annotation JSON saved to {output_json_path}"

if __name__ == "__main__":
    masks_folder = "./class_masks"
    image_path = "finetuning/outseg_pedestrians.png"
    output_json_path = "finetuning/combined_labelme.json"

    label_map = {
        0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
        5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
        9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
        14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18:"bicycle"
    }

    combine_masks_to_labelme(masks_folder, image_path, output_json_path, label_map)
