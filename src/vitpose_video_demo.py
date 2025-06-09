import cv2
from PIL import Image
from src.pose import pose_annotator, pose_processor
import logging
import numpy as np

from src.vitpose_image_demo import annotated_bgr

#initialize logger
logging.basicConfig(level=logging.DEBUG)

# Input video
video_path = "../video/slab.mp4"
captured_video = cv2.VideoCapture(video_path)

# Output video, getting parameters of original video
output_path = "../out/video/annotated_video_output.mp4"
fps = captured_video.get(cv2.CAP_PROP_FPS)

ret, frame = captured_video.read()
if not ret:
    logging.error("Couldn't read first frame!")
    exit(1)

frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE) #beacuse of recording on phone, there is a rotation flag of 90 degrees
height, width = frame.shape[:2]
captured_video.set(cv2.CAP_PROP_POS_FRAMES, 0)

# opening the writing to the output
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

logging.info(f"Reading from video {video_path}. Outputting to {output_path}, at {fps} fps, {width} width, {height} height")

if not captured_video.isOpened():
    logging.error("Error when trying to open file")

# initializing the pose processor and annotator
processor = pose_processor.PoseProcessor()
annotator = pose_annotator.PoseAnnotator()

frame_index = 1

# main loop, reading, processing, annotating and writing each frame, one by one.
while captured_video.isOpened():

    logging.debug(f"Reading frame {frame_index}")

    # reading frame
    ret, frame = captured_video.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    logging.debug(f"Converting frame {frame_index} to rgb")

    # converting frame
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    logging.debug(f"Processing frame {frame_index}")

    # processing frame
    pose_result = processor.process_frame(image)

    logging.debug(f"Done processing frame {frame_index}")

    logging.debug(f"Annotating frame {frame_index}")

    # annotating frame
    annotated_bgr = annotator.annotate_pose(image, pose_result)

    logging.debug(f"Done annotating frame {frame_index}")

    logging.debug(f"Writing frame {frame_index}")

    # writing frame
    rgb_array = np.array(image)  # shape (H, W, 3), dtype=uint8, RGB
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

    out.write(annotated_bgr)

    logging.debug(f"Done writing frame {frame_index}")

    frame_index += 1

# Clean up
captured_video.release()
out.release()
print(f"Annotated video saved to {output_path}")
