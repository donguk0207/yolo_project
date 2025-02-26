import cv2
from ultralytics import YOLO

class YoloProcessor:
    def __init__(self, model_path="yolov8n.pt"):
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        # YOLO 모델로 객체 탐지
        results = self.model(frame)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = result.names[int(box.cls[0])]
                confidence = float(box.conf[0])

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label} {confidence:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame
