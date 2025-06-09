import cv2
from PIL import Image
from src.pose import pose_annotator
from src.pose.batched import pose_processor_batched

#handling input
image_paths = ["../images/sarkany.jpg", "../images/boxolo.jpg", "../images/karate.jpg"]
images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

processor = pose_processor_batched.PoseProcessor()
pose_results = processor.process_frame(images)

annotator = pose_annotator.PoseAnnotator()

for i in range(len(images)):
    annotated_bgr = annotator.annotate_pose(images[i], pose_results[i])
    cv2.imwrite(f"../out/image/annotated_output{i + 1}.jpg", annotated_bgr)
