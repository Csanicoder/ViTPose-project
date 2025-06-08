import supervision as sv
import torch
import cv2
import numpy as np
from PIL import Image

class PoseAnnotator:

    #Annotators
    edge_annotator = sv.EdgeAnnotator(
        color=sv.Color.GREEN,
        thickness=4
    )
    vertex_annotator = sv.VertexAnnotator(
        color=sv.Color.RED,
        radius=8
    )

    def annotate_pose(self, image : Image, pose_result):

        xy = torch.stack([pose['keypoints'] for pose in pose_result]).cpu().numpy()
        scores = torch.stack([pose['scores'] for pose in pose_result]).cpu().numpy()

        key_points = sv.KeyPoints(
            xy=xy, confidence=scores
        )

        annotated_frame = self.edge_annotator.annotate(
            scene=image.copy(),
            key_points=key_points
        )
        annotated_frame = self.vertex_annotator.annotate(
            scene=annotated_frame,
            key_points=key_points
        )

        annotated_np = np.array(annotated_frame)
        annotated_bgr = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)

        return annotated_bgr
