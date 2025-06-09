import cv2
from PIL import Image

from src.pose import pose_annotator, pose_processor

# Input video
# video_path = input("Input video path: ")
video_path = "../video/slab.mp4"

captured_video = cv2.VideoCapture(video_path)

# Output video, getting parameters of original video
output_path = "../out/video/annotated_video_output.mp4"
fps = captured_video.get(cv2.CAP_PROP_FPS)
width = int(captured_video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(captured_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

# opening the writing to the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Reading from video {video_path}. Outputting to {output_path}, at {fps} fps, {width} width, {height} height")

if not captured_video.isOpened():
    print("Error when trying to open file")

# initializing the pose processor and annotator
processor = pose_processor.PoseProcessor()
annotator = pose_annotator.PoseAnnotator()

frame_index = 1

# main loop, reading, processing, annotating and writing each frame, one by one.
while captured_video.isOpened():

    print(f"Reading frame {frame_index}")
    ret, frame = captured_video.read()
    if not ret:
        break

    print(f"Converting frame {frame_index} to rgb")
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    print(f"Processing frame {frame_index}")

    pose_result = processor.process_frame(image)

    print(f"Ended processing frame {frame_index}")

    print(f"Annotating frame {frame_index}")

    annotated_bgr = annotator.annotate_pose(image, pose_result)

    print(f"Ended annotating frame {frame_index}")

    print(f"Writing frame {frame_index}")

    out.write(annotated_bgr)

    print(f"Done writing frame {frame_index}")

    frame_index += 1

# Clean up
captured_video.release()
out.release()
print(f"Annotated video saved to {output_path}")
