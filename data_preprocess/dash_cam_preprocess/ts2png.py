import os
import cv2
import ffmpeg

def extract_frames_from_ts(input_ts_file, output_folder, frame_rate=1):
    """
    Extracts frames from a .ts video file at a specified frame rate.
    
    Args:
        input_ts_file (str): Path to the input .ts file.
        output_folder (str): Directory where extracted frames will be saved.
        frame_rate (int): Number of frames to extract per second.
    """
    os.makedirs(output_folder, exist_ok=True)
    output_pattern = os.path.join(output_folder, "frame_%04d.png")

    (
        ffmpeg
        .input(input_ts_file)
        .output(output_pattern, vf=f"fps={frame_rate}")
        .run(overwrite_output=True)
    )

    print(f"Frames extracted and saved in {output_folder}")


def process_frames(input_folder, output_folder, crop_bounds, output_size=(2048, 1024)):
    """
    Crops, resizes, and saves the extracted frames.
    
    Args:
        input_folder (str): Directory containing the extracted frames.
        output_folder (str): Directory to save the processed frames.
        crop_bounds (tuple): (lower, upper, left, right) crop boundaries.
        output_size (tuple): Final resolution (width, height).
        remove_original (bool): Whether to delete the original frames after processing.
    """
    os.makedirs(output_folder, exist_ok=True)
    lower_bound, upper_bound, left_bound, right_bound = crop_bounds

    for filename in os.listdir(input_folder):
        if filename.endswith(".png"):
            image_path = os.path.join(input_folder, filename)
            image = cv2.imread(image_path)

            # Crop and resize
            cropped_image = image[lower_bound:upper_bound, left_bound:right_bound]
            resized_image = cv2.resize(cropped_image, output_size, interpolation=cv2.INTER_AREA)

            processed_path = os.path.join(output_folder, f"processed_{filename}")
            cv2.imwrite(processed_path, resized_image)

    print(f"Processed frames saved in {output_folder}")

if __name__ == "__main__":
    input_ts_file = "data/dash_cam/2025030613222033_f.ts"
    extracted_frames_folder = "data/dash_cam/frames"
    processed_frames_folder = "data/dash_cam/processed_frames"

    # Crop boundaries: (lower, upper, left, right)
    crop_bounds = (126, 1440 - 290, 140, 2560 - 372)

    extract_frames_from_ts(input_ts_file, extracted_frames_folder, frame_rate=1)
    process_frames(extracted_frames_folder, processed_frames_folder, crop_bounds)
