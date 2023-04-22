import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import mediapipe as mp
from moviepy.editor import *
from typing import Any, Tuple
from config import DataParameters
from dataclasses import dataclass
from torchvision import transforms
from architecture import AffModelClassifier, AffModelRegressor, ResNet50


@dataclass(kw_only=True)
class VideoDetector(DataParameters):

    input_video_file: str = None

    def __post_init__(self):
        self.device: torch.device = self.identify_device()
        self.cap: cv2.VideoCapture = self.get_video_stream()
        self.model_face = self.load_face_model()

    @staticmethod
    def identify_device() -> torch.device:
        """ """
        return torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def load_face_model() -> Any:
        """Function loads the yolo5 model from PyTorch Hub"""
        return mp.solutions.face_detection

    def get_video_stream(self) -> cv2.VideoCapture:
        """Function creates a streaming object to read the video from the file frame by frame"""
        if self.input_video_file is None or self.input_video_file.strip() == "":
            raise ValueError('The path to the video file cant be empty or None')

        return cv2.VideoCapture(self.input_video_file)

    def get_writer(self) -> Tuple[cv2.VideoWriter, int, int, int, int]:
        """ """
        fourcc = cv2.VideoWriter_fourcc(*'H264')

        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        img_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(os.path.join(self.output_video, 'emotions.mp4'), fourcc, fps, (img_width, img_height))

        return writer, fps, img_width, img_height, frame_count

    def load_emotional_model(self, emo_model) -> nn.Module:
        """ """
        emotion_model_load = ResNet50()
        emotion_model_load.load_state_dict(torch.load(emo_model))
        return emotion_model_load.to(self.device)

    def load_valence_model(self, val_model) -> nn.Module:
        """ """
        valence_model_load = AffModelRegressor()
        valence_model_load.load_state_dict(torch.load(val_model))
        return valence_model_load.to(self.device)

    def load_arousal_model(self, arr_model) -> nn.Module:
        """ """
        arousal_model_load = AffModelRegressor()
        arousal_model_load.load_state_dict(torch.load(arr_model))
        return arousal_model_load.to(self.device)

    def transform(self, face_image: torch.Tensor, img_size: int) -> torch.Tensor:
        """ """
        transformation = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                             ])

        face_tensor = Image.fromarray(np.uint8(face_image * 255))
        face_tensor = transformation(face_tensor).unsqueeze(0).to(self.device)
        return face_tensor

    def get_emotion_prediction(self, input_image: torch.Tensor, emo_model) -> str:
        """ """
        emotion = emo_model(input_image)
        emotion = self.emotion_dict[int(emotion.argmax(1).cpu())]
        return emotion

    @staticmethod
    def get_valence_prediction(input_image: torch.Tensor, val_model) -> float:
        """ """
        valence = round(val_model(input_image).item(), 3)
        return valence

    @staticmethod
    def get_arousal_prediction(input_image: torch.Tensor, arr_model) -> float:
        """ """
        arousal = round(arr_model(input_image).item(), 3)
        return arousal

    def get_detection(self, frame, image_width: int, image_height: int):
        """ """
        squares, coordinates = [], []

        with self.model_face.FaceDetection(min_detection_confidence=0.5, model_selection=1) as face_detection:
            results = face_detection.process(frame)
            for face_no, face in enumerate(results.detections):
                face_bbox = face.location_data.relative_bounding_box
                x1 = int(face_bbox.xmin * image_width)
                y2 = int(face_bbox.ymin * image_height)
                y1 = y2 + int(face_bbox.height * image_height)
                x2 = x1 + int(face_bbox.width * image_width)
                coordinates.append([x1, y2, x2, y1])
                square = int(x2 - x1) * int(y2 - y1)
                squares.append(square)
            max_area_ind = squares.index(max(squares))
            x1, y1, x2, y2 = coordinates[max_area_ind]

            return x1, y1, x2, y2
