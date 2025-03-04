import cv2
import numpy as np
from django.http import JsonResponse, StreamingHttpResponse, HttpResponse
from django.shortcuts import render
from .yolo import YoloProcessor
import requests
import asyncio
import threading

# ESP32 설정
ESP32_IP_LEFT = "192.168.0.20"
ESP32_IP_RIGHT = "192.168.0.19"
ESP32_STREAM_URL_LEFT = f"http://{ESP32_IP_LEFT}:81/stream"
ESP32_STREAM_URL_RIGHT = f"http://{ESP32_IP_RIGHT}:81/stream"

# YOLO 객체 탐지 프로세서
yolo_processor = YoloProcessor()

# 비동기 이벤트 루프 설정
async_loop = asyncio.new_event_loop()
asyncio.set_event_loop(async_loop)

yolo_processor_left = YoloProcessor(camera_position='left')
yolo_processor_right = YoloProcessor(camera_position='right')

def start_async_loop():
    asyncio.set_event_loop(async_loop)
    async_loop.run_forever()

# 이벤트 루프 스레드 시작
threading.Thread(target=start_async_loop, daemon=True).start()

async def async_process_frame(frame, camera):
    if camera == 'left':
        return yolo_processor_left.process_frame(frame)
    elif camera == 'right':
        return yolo_processor_right.process_frame(frame)
    else:
        raise ValueError("Invalid camera position")

def set_camera_option(request):
    camera = request.GET.get('camera', 'left')
    option = request.GET.get('option')
    value = request.GET.get('value')

    if not option or value is None:
        return JsonResponse({'status': 'error', 'message': 'Invalid parameters'}, status=400)

    ip = ESP32_IP_LEFT if camera == 'left' else ESP32_IP_RIGHT
    esp32_url = f"http://{ip}/control?var={option}&val={value}"

    try:
        response = requests.get(esp32_url, timeout=5)
        if response.status_code == 200:
            return JsonResponse({"status": "success", "message": f"{camera.capitalize()} Camera: {option} 값이 {value}로 변경되었습니다."})
        else:
            return JsonResponse({"status": "error", "message": f"{camera.capitalize()} Camera: ESP32-CAM 설정 적용 실패"}, status=500)
    except requests.exceptions.Timeout:
        return JsonResponse({"status": "error", "message": f"{camera.capitalize()} Camera: ESP32-CAM 응답 없음 (Timeout)"}, status=500)
    except requests.exceptions.RequestException as e:
        return JsonResponse({"status": "error", "message": f"{camera.capitalize()} Camera: {str(e)}"}, status=500)

def dashboard(request):
    return render(request, "dashboard.html", {
        "stream_url_left": "/yolo_stream?camera=left",
        "stream_url_right": "/yolo_stream?camera=right",
        "camera_positions": ["left", "right"]
    })

def get_esp32_stream(ip):
    try:
        return requests.get(f"http://{ip}:81/stream", stream=True, timeout=10)
    except requests.RequestException as e:
        print(f"ESP32 연결 실패 (IP: {ip}): {e}")
        return None

def generate_frames(ip, camera):
    stream = get_esp32_stream(ip)
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

            if jpg:
                try:
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is not None:
                        future = asyncio.run_coroutine_threadsafe(async_process_frame(frame, camera), async_loop)
                        processed_frame = future.result()

                        _, jpeg = cv2.imencode('.jpg', processed_frame)
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except Exception as e:
                    print(f"프레임 처리 중 오류 발생 (IP: {ip}): {e}")
            else:
                print(f"빈 JPEG 버퍼 수신 (IP: {ip})")

def yolo_stream(request):
    camera = request.GET.get('camera', 'left')
    if camera == 'left':
        ip = ESP32_IP_LEFT
    elif camera == 'right':
        ip = ESP32_IP_RIGHT
    else:
        return HttpResponse("Invalid camera", status=400)

    return StreamingHttpResponse(generate_frames(ip, camera), content_type='multipart/x-mixed-replace; boundary=frame')

def my_view(request):
    my_string = "left,right"
    camera_positions = my_string.split(',')
    return render(request, 'template.html', {'camera_positions': camera_positions})
