import csv
import time
from datetime import datetime
import numpy as np
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from picamera2 import Picamera2
from caracteristicas import caracteristicas

UMBRAL = 0.85        # confianza mínima para aceptar a alguien
CONFIRMAR = 2        # frames seguidos con la misma persona para marcar presente

modelo = joblib.load("modelo.joblib")
lista = list(modelo.classes_)
presentes = {}       
racha_nombre, racha = None, 0

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

print("Q = terminar y guardar asistencia")
try:
    while True:
        frame = cam.capture_array()
        rgb = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        texto, color = "SIN CARA", (0, 0, 255)
        if res.face_landmarks:
            puntos = np.array([[p.x, p.y, p.z] for p in res.face_landmarks[0]], dtype=np.float32)
            probs = modelo.predict_proba([caracteristicas(puntos, usar_z=True)])[0]
            i = int(np.argmax(probs))
            nombre, conf = lista[i], probs[i]

            if conf >= UMBRAL:
                texto, color = f"{nombre} ({conf * 100:.0f}%)", (0, 255, 0)
                racha = racha + 1 if nombre == racha_nombre else 1
                racha_nombre = nombre
                if racha >= CONFIRMAR and nombre not in presentes:
                    presentes[nombre] = datetime.now().strftime("%H:%M:%S")
                    print(f"PRESENTE: {nombre} a las {presentes[nombre]}")
            else:
                texto, color = f"Desconocido ({conf * 100:.0f}%)", (0, 165, 255)
                racha_nombre, racha = None, 0
        else:
            racha_nombre, racha = None, 0

        cv2.putText(frame, texto, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Presentes: {len(presentes)}/{len(lista)}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Pase de lista", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break
finally:
    cam.stop()
    cv2.destroyAllWindows()
    archivo = f"asistencia_{datetime.now():%Y-%m-%d_%H%M}.csv"
    with open(archivo, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["nombre", "estado", "hora"])
        for n in lista:
            w.writerow([n, "presente" if n in presentes else "ausente", presentes.get(n, "")])
    print(f"Asistencia guardada en {archivo}")
