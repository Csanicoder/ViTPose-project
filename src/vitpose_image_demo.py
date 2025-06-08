from PIL import Image
import cv2
from src.pose import pose_annotator, pose_processor
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

#handling input
image_path = "../images/sarkany.jpg"

logger.debug("Opening image")

image = Image.open(image_path).convert("RGB")

logger.debug("Done opening image")

logger.debug("Processing image")

processor = pose_processor.PoseProcessor()
image_pose_result = processor.process_frame(image)

logger.debug("Done processing image")

logger.debug("Annotating image")

annotator = pose_annotator.PoseAnnotator()
annotated_bgr = annotator.annotate_pose(image, image_pose_result)

logger.debug("Done annotating image")

cv2.imwrite('../out/image/annotated_output.jpg', annotated_bgr)