import cv2
import os

# Directories for original and generated frames
original_frames_dir = "mydata/scene1/aps_png"
generated_frames_dir = "RGB_MyInferenceDataset/MyInferenceDataset_Expv8_large_infer_pre_x4_scene1/Validation_Visual_Examples/images/scene_scene1/epoch_10"

# Video output settings
output_video_path = "pre_output_video1_40fps.mp4"
fps = 40  # Specify the frame rate
frame_size = (816, 612)  # Modify this according to your frame resolution

# Initialize video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

# Read original and generated frames
frame_count = len([f for f in os.listdir(original_frames_dir) if f.endswith('.png')])

for i in range(frame_count - 1):  # Loop over the frame indices
    # Original frames (e.g., 000000.png, 000001.png, etc.)
    original_frame_path = os.path.join(original_frames_dir, f"{str(i).zfill(6)}.png")
    original_frame = cv2.imread(original_frame_path)
    
    # Generated frames (e.g., frames 000000_to_000001_t0.0000.png, 000000_to_000001_t0.3333.png)
    generated_frame_1 = cv2.imread(os.path.join(generated_frames_dir, f"{str(i).zfill(6)}_to_{str(i+1).zfill(6)}_t0.0000.png"))
    generated_frame_2 = cv2.imread(os.path.join(generated_frames_dir, f"{str(i).zfill(6)}_to_{str(i+1).zfill(6)}_t0.3333.png"))
    generated_frame_3 = cv2.imread(os.path.join(generated_frames_dir, f"{str(i).zfill(6)}_to_{str(i+1).zfill(6)}_t0.6667.png"))
    
    # Writing the frames to video, alternating original and generated frames
    out.write(original_frame)  # Write original frame
    out.write(generated_frame_1)  # Write generated frame
    out.write(generated_frame_2)  # Write generated frame
    out.write(generated_frame_3)  # Write generated frame

# Release video writer
out.release()
cv2.destroyAllWindows()

print("Video creation completed.")
