import cv2
import numpy as np
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from .yolo import YoloProcessor
import requests
import asyncio
import threading

# ESP32 설정
ESP32_IP = "192.168.0.19"
ESP32_STREAM_URL = f"http://{ESP32_IP}:81/stream"

# YOLO 객체 탐지 프로세서
yolo_processor = YoloProcessor()

# 비동기 이벤트 루프 설정
async_loop = asyncio.new_event_loop()
asyncio.set_event_loop(async_loop)

def start_async_loop():
    asyncio.set_event_loop(async_loop)
    async_loop.run_forever()

# 이벤트 루프 스레드 시작
threading.Thread(target=start_async_loop, daemon=True).start()

async def async_process_frame(frame):
    return yolo_processor.process_frame(frame)

def set_camera_option(request):
    option = request.GET.get('option')
    value = request.GET.get('value')

    if not option or value is None:
        return JsonResponse({'status': 'error', 'message': 'Invalid parameters'}, status=400)

    esp32_url = f"http://{ESP32_IP}/control?var={option}&val={value}"

    try:
        response = requests.get(esp32_url, timeout=5)
        if response.status_code == 200:
            return JsonResponse({"status": "success", "message": f"{option} 값이 {value}로 변경되었습니다."})
        else:
            return JsonResponse({"status": "error", "message": "ESP32-CAM 설정 적용 실패"}, status=500)
    except requests.exceptions.Timeout:
        return JsonResponse({"status": "error", "message": "ESP32-CAM 응답 없음 (Timeout)"}, status=500)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"status": "error", "message": str(e)}, status=500)

def dashboard(request):
    return render(request, "dashboard.html", {"stream_url": "/yolo_stream"})

def get_esp32_stream():
    try:
        return requests.get(ESP32_STREAM_URL, stream=True, timeout=10)
    except requests.RequestException as e:
        print(f"ESP32 연결 실패: {e}")
        return None

def generate_frames():
    stream = get_esp32_stream()
    if not stream:
        return

    bytes_data = b""
    while True:
        chunk = stream.raw.read(1024)
        if not chunk:
            break
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')
        b = bytes_data.find(b'\xff\xd9')

        if a != -1 and b != -1:
            jpg = bytes_data[a:b + 2]
            bytes_data = bytes_data[b + 2:]

            if jpg:  # 빈 버퍼 체크 추가
                try:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        future = asyncio.run_coroutine_threadsafe(async_process_frame(frame), async_loop)
                        processed_frame = future.result()

                        _, jpeg = cv2.imencode('.jpg', processed_frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except Exception as e:
                    print(f"프레임 처리 중 오류 발생: {e}")
            else:
                print("빈 JPEG 버퍼 수신")

def yolo_stream(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def yolo_stream(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')
