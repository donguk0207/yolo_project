import cv2
import numpy as np
from ultralytics import YOLO

# 모델 경로: "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt" 중 선택
# 신뢰도 임계값: 0.1 ~ 1.0 사이의 값 (높을수록 엄격한 탐지)
# NMS(비최대 억제) 임계값: 0.1 ~ 1.0 사이의 값 (높을수록 중복 탐지 허용)
# 배치 크기: 1 이상의 정수 (큰 값은 처리 속도를 높이지만 메모리 사용량 증가)  batch_size: 1 이상의 정수
# 학습률: 0.0001 ~ 0.1 사이의 값 (높은 값은 빠른 학습, 낮은 값은 안정적인 학습) learning_rate: 0.0001 ~ 0.1 사이의 값
# 모멘텀: 0 ~ 1 사이의 값 (높은 값은 이전 그래디언트의 영향을 더 많이 받음) momentum: 0 ~ 1 사이의 값
# 이미지 크기: 32의 배수 (큰 값은 정확도 향상, 작은 값은 속도 향상) image_size: 32의 배수 (예: 320, 416, 512, 608, 640 등)

class YoloProcessor:
    def __init__(self, camera_position=None, model_path="yolov8n.pt", conf_threshold=0.8, nms_threshold=0.4,
                 batch_size=1, learning_rate=0.01, momentum=0.937, image_size=640):
        self.camera_position = camera_position
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.classes_to_detect = None

        # 하이퍼파라미터 설정
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.momentum = momentum

        # 이미지 크기 설정
        self.image_size = image_size

    def process_frame(self, frame):
        # 이미지 크기 조정
        processed_frame = cv2.resize(frame, (self.image_size, self.image_size))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        # 배치 처리를 위해 차원 추가
        processed_frame = np.expand_dims(processed_frame, axis=0)

        # YOLO 모델로 객체 탐지 (배치 크기 적용)
        results = self.model(processed_frame, conf=self.conf_threshold, iou=self.nms_threshold,
                             batch=self.batch_size)

        result = results[0]
        boxes = result.boxes.cpu().numpy()

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            if self.classes_to_detect is None or class_id in self.classes_to_detect:
                label = f"{result.names[class_id]} {confidence:.2f}"

                # 카메라 위치에 따라 색상 변경
                if self.camera_position == 'left':
                    color = (0, 255, 0)  # 왼쪽 카메라: 녹색
                elif self.camera_position == 'right':
                    color = (0, 0, 255)  # 오른쪽 카메라: 빨간색
                else:
                    color = (255, 0, 0)  # 기본: 파란색

                # 원본 프레임 크기에 맞게 바운딩 박스 좌표 조정
                h, w = frame.shape[:2]
                x1, y1 = int(x1 * w / self.image_size), int(y1 * h / self.image_size)
                x2, y2 = int(x2 * w / self.image_size), int(y2 * h / self.image_size)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def process_frame(self, frame):
        processed_frame = cv2.resize(frame, (640, 640))
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        results = self.model(processed_frame, conf=self.conf_threshold, iou=self.nms_threshold)

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if self.classes_to_detect is None or class_id in self.classes_to_detect:
                    label = f"{result.names[class_id]} {confidence:.2f}"

                    # 카메라 위치에 따라 색상 변경
                    if self.camera_position == 'left':
                        color = (0, 255, 0)  # 왼쪽 카메라: 녹색
                    elif self.camera_position == 'right':
                        color = (0, 0, 255)  # 오른쪽 카메라: 빨간색
                    else:
                        color = (255, 0, 0)  # 기본: 파란색

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
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

    def set_hyperparameters(self, batch_size=None, learning_rate=None, momentum=None):
        """
        하이퍼파라미터 설정
        """
        if batch_size is not None:
            self.batch_size = batch_size
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if momentum is not None:
            self.momentum = momentum

    def set_image_size(self, image_size):
        """
        이미지 크기 설정
        """
        self.image_size = image_size
