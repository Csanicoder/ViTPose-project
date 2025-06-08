import cv2
from PIL import Image
from src.pose import pose_processor_batched, pose_annotator
import numpy as np

#--------------------------------------------------------------------
# Input video
video_path = "../video/slab.mp4"
cap = cv2.VideoCapture(video_path)

# Output video, getting parameters of original video
output_path = "../out/video/annotated_video_output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Reading from video {video_path}. Outputting to {output_path}, at {fps} fps, {width} width, {height} height")

#--------------------------------------------------------------------
#actual processing
batch_size = 16

processor = pose_processor_batched.PoseProcessor()
annotator = pose_annotator.PoseAnnotator()

batch_index = 1

while cap.isOpened():
    print(f"Reading batch {batch_index}")
    frames = []
    for _ in range(batch_size):
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(rgb))

    if not frames:
        break

    print(f"Done reading batch {batch_index}")

    print(f"Processing batch {batch_index}")

    pose_results = processor.process_frame(frames)

    print(f"Done processing batch {batch_index}")

    print(f"Writing batch {batch_index}")

    for i in range(len(frames)):
        annotated_bgr = annotator.annotate_pose(frames[i], pose_results[i])
        out.write(annotated_bgr)

    print(f"Done writing batch {batch_index}")

    batch_index += 1

print(f"Finished proccesing {video_path}, wrote results to {output_path}")

cap.release()
out.release()