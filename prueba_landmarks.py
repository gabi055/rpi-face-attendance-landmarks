from picamera2 import Picamera2
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

opciones = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(opciones)

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()

frame = cam.capture_array()
cam.stop()

imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
resultado = detector.detect(imagen)

if resultado.face_landmarks:
    puntos = resultado.face_landmarks[0]
    print("Puntos detectados:", len(puntos))
    h, w = frame.shape[:2]
    for p in puntos:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 1, (0, 255, 0), -1)
else:
    print("No se detectó cara")

cv2.imwrite("landmarks.jpg", frame)
