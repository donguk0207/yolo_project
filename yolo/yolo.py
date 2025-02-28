import cv2
import numpy as np
from ultralytics import YOLO


class YoloProcessor:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.8, nms_threshold=0.2):
        # 모델 경로: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt" 중 선택
        self.model = YOLO(model_path)

        # 신뢰도 임계값: 0.1 ~ 1.0 사이의 값 (높을수록 엄격한 탐지)
        self.conf_threshold = conf_threshold

        # NMS(비최대 억제) 임계값: 0.1 ~ 1.0 사이의 값 (높을수록 중복 탐지 허용)
        self.nms_threshold = nms_threshold

        # 탐지할 클래스 목록: None이면 모든 클래스 탐지, 리스트로 지정하면 특정 클래스만 탐지
        self.classes_to_detect = None

    def process_frame(self, frame):
        # 이미지 전처리
        # 크기 조정: (width, height) 형식으로 변경 가능, 예: (320, 320), (416, 416), (608, 608)
        processed_frame = cv2.resize(frame, (640, 640))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # YOLO 모델로 객체 탐지
        results = self.model(processed_frame, conf=self.conf_threshold, iou=self.nms_threshold)

        # 결과 처리 및 시각화
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if self.classes_to_detect is None or class_id in self.classes_to_detect:
                    label = f"{result.names[class_id]} {confidence:.2f}"

                    # 색상 설정: (B, G, R) 형식, 0-255 사이의 값
                    color = (0, 255, 0)  # 기본 녹색, 원하는 색상으로 변경 가능

                    # 바운딩 박스 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    # 텍스트 설정: 폰트, 크기, 색상 등을 변경 가능
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def set_classes_to_detect(self, classes):
        """
        특정 클래스만 탐지하도록 설정
        :param classes: 탐지할 클래스 ID 리스트, 예: [0, 1, 2] (사람, 자동차, 개)
        """
        self.classes_to_detect = classes

    def set_conf_threshold(self, threshold):
        """
        신뢰도 임계값 설정
        :param threshold: 0.1 ~ 1.0 사이의 값
        """
        self.conf_threshold = max(0.1, min(threshold, 1.0))

    def set_nms_threshold(self, threshold):
        """
        NMS 임계값 설정
        :param threshold: 0.1 ~ 1.0 사이의 값
        """
        self.nms_threshold = max(0.1, min(threshold, 1.0))
