import numpy as np

ANCHO, ALTO = 640, 480      # resolución con la que capturaste
OJO_D, OJO_I = 33, 263      # esquinas externas de los ojos (referencia)

# ~50 puntos esenciales: ojos, cejas, nariz, boca y contorno
INDICES = [
    33, 133, 159, 145, 160, 144,        # ojo derecho
    263, 362, 386, 374, 385, 380,       # ojo izquierdo
    70, 63, 105, 66, 107,               # ceja derecha
    300, 293, 334, 296, 336,            # ceja izquierda
    1, 4, 6, 168, 98, 327, 129, 358,    # nariz
    61, 291, 13, 14, 0, 17, 37, 267, 84, 314,   # boca
    10, 152, 234, 454, 127, 356, 172, 397, 58, 288,  # contorno
]

def normalizar(puntos):
    """puntos: arreglo (478, 3) crudo de MediaPipe."""
    p = puntos.astype(np.float64).copy()
    # MediaPipe da valores entre 0 y 1: pasamos a píxeles para no deformar la cara
    p[:, 0] *= ANCHO
    p[:, 1] *= ALTO
    p[:, 2] *= ANCHO
    # 1) centrar en el punto medio entre los ojos
    p -= (p[OJO_D] + p[OJO_I]) / 2
    # 2) rotar para que la línea de los ojos quede horizontal (corrige roll)
    dx = p[OJO_I, 0] - p[OJO_D, 0]
    dy = p[OJO_I, 1] - p[OJO_D, 1]
    ang = np.arctan2(dy, dx)
    c, s = np.cos(-ang), np.sin(-ang)
    R = np.array([[c, -s], [s, c]])
    p[:, :2] = p[:, :2] @ R.T
    # 3) escalar por la distancia entre ojos (corrige distancia a la cámara)
    p /= np.hypot(dx, dy)
    return p

def caracteristicas(puntos, usar_z=False):
    p = normalizar(puntos)[INDICES]
    if not usar_z:
        p = p[:, :2]        # solo x, y (z de MediaPipe es más ruidosa)
    return p.flatten()
