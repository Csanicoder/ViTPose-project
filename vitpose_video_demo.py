import torch
import numpy as np
import cv2
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
import supervision as sv

# Input video
#video_path = input("Input video path: ")
video_path = "video/campus.mp4"

cap = cv2.VideoCapture(video_path)

# Output video
output_path = "annotated_video_output.mp4"
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Reading from video {video_path}. Outputting to {output_path}, at {fps} fps, {width} width, {height} height")

#setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

#load models
person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", use_fast = True)
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

image_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple", use_fast = True)
model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

#Annotators
edge_annotator = sv.EdgeAnnotator(
    color=sv.Color.GREEN,
    thickness=4
)
vertex_annotator = sv.VertexAnnotator(
    color=sv.Color.RED,
    radius=8
)

def proccess_frame(image):
    # ------------------------------------------------------------------------
    # Stage 1. Detect humans on the image
    # ------------------------------------------------------------------------

    inputs = person_image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.3
    )
    result = results[0]  # take first image result

    # Human label refers 0 index in COCO dataset
    person_boxes = result["boxes"][result["labels"] == 0]
    person_boxes = person_boxes.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
    person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
    person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

    # ------------------------------------------------------------------------
    # Stage 2. Detect keypoints for each person found
    # ------------------------------------------------------------------------

    inputs = image_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    pose_results = image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])
    image_pose_result = pose_results[0]  # results for first image

    xy = torch.stack([pose_result['keypoints'] for pose_result in image_pose_result]).cpu().numpy()
    scores = torch.stack([pose_result['scores'] for pose_result in image_pose_result]).cpu().numpy()

    key_points = sv.KeyPoints(
        xy=xy, confidence=scores
    )

    annotated_frame = edge_annotator.annotate(
        scene=image.copy(),
        key_points=key_points
    )
    annotated_frame = vertex_annotator.annotate(
        scene=annotated_frame,
        key_points=key_points
    )

    annotated_np = np.array(annotated_frame)
    annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)

    out.write(annotated_bgr)

if not cap.isOpened():
    print("Error when trying to open file")

frame_index = 1

while cap.isOpened():
    print(f"Reading frame {frame_index}")
    ret, frame = cap.read()
    if not ret:
        break

    print(f"Converting frame {frame_index}")

    # Convert to PIL image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    print(f"Proccessing frame {frame_index}")

    proccess_frame(image)

    print(f"Ended proccessing frame {frame_index}")

    frame_index += 1

# Clean up
cap.release()
out.release()
print(f"Annotated video saved to {output_path}")
