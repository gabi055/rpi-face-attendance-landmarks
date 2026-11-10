import os
import time
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from picamera2 import Picamera2

META = 40               # muestras objetivo por persona
INTERVALO_AUTO = 0.5    # segundos entre muestras en modo automático

nombre = input("Nombre de la persona: ").strip().replace(" ", "_")
carpeta = os.path.join("dataset", nombre)
os.makedirs(carpeta, exist_ok=True)
total = len([f for f in os.listdir(carpeta) if f.endswith(".npy")])

opciones = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path="face_landmarker.task"),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
detector = vision.FaceLandmarker.create_from_options(opciones)

cam = Picamera2()
cam.configure(cam.create_preview_configuration(main={"size": (640, 480), "format": "RGB888"}))
cam.start()
time.sleep(1)  

auto = False
ultimo_guardado = 0

print("ESPACIO = guardar una muestra | C = modo automático on/off | Q = salir")

while True:
    frame = cam.capture_array()  
    rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    resultado = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    hay_cara = bool(resultado.face_landmarks)
    puntos = None
    if hay_cara:
        puntos = np.array([[p.x, p.y, p.z] for p in resultado.face_landmarks[0]], dtype=np.float32)
        h, w = frame.shape[:2]
        for x, y, _ in puntos[::4]:  
            cv2.circle(frame, (int(x * w), int(y * h)), 1, (0, 255, 0), -1)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla in (ord("q"), 27):
        break
    if tecla == ord("c"):
        auto = not auto

    guardar = False
    if hay_cara:
        if tecla == 32:
            guardar = True
        elif auto and time.time() - ultimo_guardado >= INTERVALO_AUTO:
            guardar = True

    if guardar:
        archivo = os.path.join(carpeta, f"{nombre}_{int(time.time() * 1000)}.npy")
        np.save(archivo, puntos)
        total += 1
        ultimo_guardado = time.time()

    estado = "CARA OK" if hay_cara else "SIN CARA"
    modo = "AUTO" if auto else "MANUAL"
    cv2.putText(frame, f"{nombre} | {total}/{META} | {modo} | {estado}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imshow("Captura", frame)

    if total >= META:
        print("Meta alcanzada.")
        break

cam.stop()
cv2.destroyAllWindows()
print(f"Muestras guardadas para {nombre}: {total}")
