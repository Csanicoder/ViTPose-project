import torch
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
import os

class PoseProcessor:
    """
    This class is used to process the people visible on an image, and identify their keypoints (shoulders, hips, knees, etc.)
    """

    def __init__(self, local_person_path = "/home/csand/.cache/huggingface/hub/models--PekingU--rtdetr_r50vd_coco_o365/snapshots/457857cec8ac28ddede40ecee9eed2beca321af8"
                 , local_path = "/home/csand/.cache/huggingface/hub/models--usyd-community--vitpose-base-simple/snapshots/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45"):
        """
        initialize class attributes
        :param local_person_path: path to local huggingface object detection models - default is RT-DETR
        :param local_path: path to local huggingface pose estimation models - default is ViTPose
        """
        # setup device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # load models
        self.person_image_processor = AutoProcessor.from_pretrained(local_person_path, use_fast = True)
        self.person_model = RTDetrForObjectDetection.from_pretrained(local_person_path, device_map=self.device)

        self.image_processor = AutoProcessor.from_pretrained(local_path, use_fast = True)
        self.model = VitPoseForPoseEstimation.from_pretrained(local_path, device_map=self.device)

    # main method, used for processing now multiple images and outputting the pose result
    def process_frame(self, image):

        # converting images into the format that the model needs
        inputs = self.person_image_processor(images=image, return_tensors="pt").to(self.device)

        # inference with the model
        with torch.no_grad():
            outputs = self.person_model(**inputs)

        # reconverting output into the image format
        results = self.person_image_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([(image.height, image.width)]), # converting normalized coordinates back to the dimensions of the image
            threshold=0.3 # ignoring boxes under a score of 0.3
        )
        result = results[0]

        # Human label refers 0 index in COCO dataset
        person_boxes = result["boxes"][result["labels"] == 0]
        person_boxes = person_boxes.cpu().numpy()

        # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) format
        person_boxes[:, 2] = person_boxes[:, 2] - person_boxes[:, 0]
        person_boxes[:, 3] = person_boxes[:, 3] - person_boxes[:, 1]

       # detect keypoints, mostly same as before
        inputs = self.image_processor(image, boxes=[person_boxes], return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # results for first image
        pose_results = self.image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])

        return pose_results[0]
