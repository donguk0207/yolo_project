import cv2
from ultralytics import YOLO

class YoloProcessor:
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        # YOLO 모델로 객체 탐지

        for result in results:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])


        return frame
