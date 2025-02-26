from django.urls import path
from . import views

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
    #path('stream_video', views.stream_video, name='stream_video'),
    path('set_camera_option/', views.set_camera_option, name='set_camera_option'),  # 슬래시 추가
    path('yolo_stream/', views.yolo_stream, name='yolo_stream'),
]