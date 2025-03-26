import json
import os

def convert_labelme_to_cityscapes(labelme_json_path, output_path):
    with open(labelme_json_path, 'r') as f:
        data = json.load(f)

    converted = {
        "imgHeight": data["imageHeight"],
        "imgWidth": data["imageWidth"],
        "objects": []
    }

    for shape in data["shapes"]:
        int_polygon = [[int(x), int(y)] for x, y in shape["points"]]
        obj = {
            "label": shape["label"],
            "polygon": int_polygon
        }
        converted["objects"].append(obj)

    with open(output_path, 'w') as f:
        json.dump(converted, f, indent=2)


# Example usage
convert_labelme_to_cityscapes(
    labelme_json_path="processed_frame_0002.json",
    output_path="processed_frame_0002_gtFine_polygons.json"
)
