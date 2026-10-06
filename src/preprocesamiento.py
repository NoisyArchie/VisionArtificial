"""
preprocesamiento.py
Pipeline de preprocesamiento de video con OpenCV.

Nota de diseño: Ultralytics YOLO ya aplica internamente letterbox, BGR->RGB y
normalización 0-1. Por eso el Preprocesador NO repite esos pasos antes de la
inferencia; se encarga de lo que el modelo no hace solo:
  - corrección de iluminación (CLAHE y gamma automática para escenas oscuras)
  - región de interés (ROI) para ignorar zonas del cuadro que no importan
  - reducción de resolución para ganar FPS
Las funciones letterbox() y normalizar() se incluyen para documentar y
visualizar lo que sucede dentro del modelo (sirve para el informe técnico).
"""

import time

import cv2
import numpy as np


# ---------------------------------------------------------------- utilidades
def letterbox(img, tam=640, color=(114, 114, 114)):
    """Redimensiona sin deformar y rellena con bordes hasta tam x tam.
    Regresa (imagen, escala, (relleno_x, relleno_y))."""
    alto, ancho = img.shape[:2]
    r = min(tam / alto, tam / ancho)
    nuevo_ancho, nuevo_alto = int(round(ancho * r)), int(round(alto * r))
    redim = cv2.resize(img, (nuevo_ancho, nuevo_alto), interpolation=cv2.INTER_LINEAR)
    arriba = (tam - nuevo_alto) // 2
    izquierda = (tam - nuevo_ancho) // 2
    salida = cv2.copyMakeBorder(
        redim, arriba, tam - nuevo_alto - arriba, izquierda, tam - nuevo_ancho - izquierda,
        cv2.BORDER_CONSTANT, value=color,
    )
    return salida, r, (izquierda, arriba)


def normalizar(img):
    """BGR uint8 (0-255) -> RGB float32 (0-1). Equivale a lo que hace YOLO por dentro."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def brillo_medio(img):
    """Brillo promedio (0-255) usando el canal de luminancia."""
    return float(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).mean())


def aplicar_clahe(img, clip=2.0, rejilla=8):
    """Ecualización adaptativa de histograma sobre el canal L (LAB).
    Mejora el contraste local sin alterar los colores."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(rejilla, rejilla))
    return cv2.cvtColor(cv2.merge((clahe.apply(l), a, b)), cv2.COLOR_LAB2BGR)


def correccion_gamma(img, gamma):
    """Aplica salida = (entrada/255)^gamma * 255. gamma < 1 aclara, gamma > 1 oscurece."""
    tabla = (np.linspace(0, 1, 256) ** gamma * 255).clip(0, 255).astype(np.uint8)
    return cv2.LUT(img, tabla)


def gamma_automatica(img, objetivo=128, minimo=0.4):
    """Calcula la gamma que lleva el brillo medio hacia 'objetivo'. Solo aclara."""
    media = max(brillo_medio(img), 1.0)
    if media >= objetivo:
        return 1.0
    gamma = np.log(objetivo / 255) / np.log(media / 255)
    return float(np.clip(gamma, minimo, 1.0))


def construir_mascara_roi(forma, poligono):
    """Máscara binaria a partir de un polígono [(x, y), ...] en píxeles."""
    mascara = np.zeros(forma[:2], dtype=np.uint8)
    cv2.fillPoly(mascara, [np.array(poligono, dtype=np.int32)], 255)
    return mascara


def aplicar_roi(img, mascara):
    """Pone en negro todo lo que está fuera de la región de interés."""
    return cv2.bitwise_and(img, img, mask=mascara)


def redimensionar_max(img, lado_max):
    """Reduce la imagen para que su lado mayor no pase de lado_max (no agranda)."""
    alto, ancho = img.shape[:2]
    escala = lado_max / max(alto, ancho)
    if escala >= 1:
        return img, 1.0
    return cv2.resize(img, (int(ancho * escala), int(alto * escala)),
                      interpolation=cv2.INTER_AREA), escala


# ---------------------------------------------------------------- pipeline
class Preprocesador:
    """
    Pipeline configurable que se aplica a cada cuadro antes de pasarlo a YOLO.

    lado_max          : None o entero; reduce la resolución de entrada
    clahe             : 'auto' (solo en escenas oscuras), True (siempre) o False
    gamma             : True para aplicar gamma automática en escenas oscuras
    umbral_oscuridad  : brillo medio (0-255) por debajo del cual la escena se considera oscura
    roi               : None o lista de puntos [(x, y), ...] en coordenadas del cuadro ya redimensionado
    """

    def __init__(self, lado_max=None, clahe="auto", gamma=True, umbral_oscuridad=80,
                 clip=2.0, roi=None):
        self.lado_max = lado_max
        self.clahe = clahe
        self.gamma = gamma
        self.umbral_oscuridad = umbral_oscuridad
        self.clip = clip
        self.roi = roi
        self._mascara = None
        self.tiempos_ms = []
        self.ultima_info = {}

    def __call__(self, frame):
        t0 = time.perf_counter()
        info = {}

        if self.lado_max:
            frame, info["escala"] = redimensionar_max(frame, self.lado_max)

        brillo = brillo_medio(frame)
        oscura = brillo < self.umbral_oscuridad
        info["brillo"] = round(brillo, 1)
        info["oscura"] = oscura

        if self.gamma and oscura:
            g = gamma_automatica(frame)
            frame = correccion_gamma(frame, g)
            info["gamma"] = round(g, 3)

        if self.clahe is True or (self.clahe == "auto" and oscura):
            frame = aplicar_clahe(frame, clip=self.clip)
            info["clahe"] = True

        if self.roi is not None:
            if self._mascara is None or self._mascara.shape[:2] != frame.shape[:2]:
                self._mascara = construir_mascara_roi(frame.shape, self.roi)
            frame = aplicar_roi(frame, self._mascara)

        self.tiempos_ms.append((time.perf_counter() - t0) * 1000)
        self.ultima_info = info
        return frame

    def resumen_tiempos(self):
        if not self.tiempos_ms:
            return {}
        t = np.array(self.tiempos_ms)
        return {"cuadros": len(t), "promedio_ms": round(float(t.mean()), 2),
                "p95_ms": round(float(np.percentile(t, 95)), 2)}


# ---------------------------------------------------------------- video
def iterar_video(ruta, saltar=0):
    """Generador de cuadros de un video. saltar=n procesa 1 de cada n+1 cuadros."""
    cap = cv2.VideoCapture(str(ruta))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {ruta}")
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if saltar == 0 or i % (saltar + 1) == 0:
                yield i, frame
            i += 1
    finally:
        cap.release()


def info_video(ruta):
    cap = cv2.VideoCapture(str(ruta))
    datos = {
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "cuadros": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "ancho": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "alto": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return datos


def simular_oscuridad(img, factor=0.35):
    """Oscurece una imagen para probar el pipeline sin necesitar video nocturno."""
    return (img.astype(np.float32) * factor).clip(0, 255).astype(np.uint8)
