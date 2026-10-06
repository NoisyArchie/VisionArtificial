"""
conversion_datos.py
Conversión de las anotaciones de CrowdHuman (.odgt) al formato que usa Ultralytics YOLO.

Formato .odgt: cada línea es un JSON con las anotaciones de una imagen:
{"ID": "273271,c9db000d5146c15",
 "gtboxes": [{"tag": "person", "fbox": [x, y, w, h], "vbox": [...], "hbox": [...],
              "extra": {"ignore": 0}, "head_attr": {"ignore": 0}}, ...]}

Formato YOLO: un .txt por imagen, una línea por objeto:
<clase> <x_centro> <y_centro> <ancho> <alto>   (valores normalizados entre 0 y 1)
"""

import json
import random
import shutil
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image

TIPOS_CAJA = {
    "vbox": "cuerpo visible",
    "fbox": "cuerpo completo",
    "hbox": "cabeza",
}


def leer_odgt(ruta):
    """Lee un archivo .odgt y regresa un diccionario por imagen (una línea = un JSON)."""
    with open(ruta, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                yield json.loads(linea)


def es_ignorada(gt, tipo_caja):
    """True si la anotación no debe usarse (no es persona o está marcada como 'ignore')."""
    if gt.get("tag") != "person":  # 'mask' = maniquíes, reflejos, carteles, etc.
        return True
    if gt.get("extra", {}).get("ignore", 0) == 1:
        return True
    if tipo_caja == "hbox" and gt.get("head_attr", {}).get("ignore", 0) == 1:
        return True
    return False


def caja_a_yolo(caja, ancho_img, alto_img):
    """Convierte [x, y, w, h] en píxeles a (xc, yc, w, h) normalizados.
    Recorta la caja a los bordes de la imagen (fbox puede salirse de la imagen)."""
    x, y, w, h = caja
    x1, y1 = max(0.0, x), max(0.0, y)
    x2, y2 = min(float(ancho_img), x + w), min(float(alto_img), y + h)
    w_rec, h_rec = x2 - x1, y2 - y1
    if w_rec <= 0 or h_rec <= 0:
        return None
    return (
        (x1 + w_rec / 2) / ancho_img,
        (y1 + h_rec / 2) / alto_img,
        w_rec / ancho_img,
        h_rec / alto_img,
        w_rec,
        h_rec,
    )


def indexar_imagenes(carpetas):
    """Busca todas las .jpg dentro de las carpetas y regresa {ID: ruta}."""
    indice = {}
    for carpeta in carpetas:
        for ruta in Path(carpeta).rglob("*.jpg"):
            indice[ruta.stem] = ruta
    return indice


def analizar_odgt(ruta_odgt):
    """Estadísticas rápidas del dataset para el reporte."""
    personas_por_imagen = []
    etiquetas = Counter()
    ignoradas = 0
    for registro in leer_odgt(ruta_odgt):
        n = 0
        for gt in registro.get("gtboxes", []):
            etiquetas[gt.get("tag")] += 1
            if gt.get("tag") == "person" and gt.get("extra", {}).get("ignore", 0) == 0:
                n += 1
            else:
                ignoradas += 1
        personas_por_imagen.append(n)
    total = len(personas_por_imagen)
    return {
        "imagenes": total,
        "personas": sum(personas_por_imagen),
        "promedio_por_imagen": sum(personas_por_imagen) / total if total else 0,
        "maximo_por_imagen": max(personas_por_imagen) if total else 0,
        "anotaciones_ignoradas": ignoradas,
        "etiquetas": dict(etiquetas),
        "personas_por_imagen": personas_por_imagen,
    }


def convertir_split(ruta_odgt, carpetas_imagenes, destino, split,
                    tipo_caja="vbox", min_lado_px=4, mover=False, limite=None):
    """
    Convierte un split (train o val) de CrowdHuman a formato YOLO.

    ruta_odgt         : annotation_train.odgt o annotation_val.odgt
    carpetas_imagenes : lista de carpetas donde están las .jpg descomprimidas
    destino           : carpeta raíz del dataset YOLO (se crean images/<split> y labels/<split>)
    tipo_caja         : 'vbox' (visible), 'fbox' (completo) o 'hbox' (cabeza)
    min_lado_px       : descarta cajas con ancho o alto menor a esto (ruido)
    mover             : True mueve las imágenes en lugar de copiarlas (ahorra disco en Colab)
    limite            : número máximo de imágenes (útil para pruebas rápidas)
    """
    if tipo_caja not in TIPOS_CAJA:
        raise ValueError(f"tipo_caja debe ser uno de {list(TIPOS_CAJA)}")

    destino = Path(destino)
    dir_img = destino / "images" / split
    dir_lbl = destino / "labels" / split
    dir_img.mkdir(parents=True, exist_ok=True)
    dir_lbl.mkdir(parents=True, exist_ok=True)

    indice = indexar_imagenes(carpetas_imagenes)
    stats = Counter()

    for registro in leer_odgt(ruta_odgt):
        if limite is not None and stats["imagenes"] >= limite:
            break
        id_img = registro["ID"]
        ruta_img = indice.get(id_img)
        if ruta_img is None:
            stats["sin_imagen"] += 1  # pasa si solo descomprimiste una parte del train
            continue

        with Image.open(ruta_img) as im:
            ancho, alto = im.size

        lineas = []
        for gt in registro.get("gtboxes", []):
            if es_ignorada(gt, tipo_caja) or tipo_caja not in gt:
                stats["cajas_ignoradas"] += 1
                continue
            res = caja_a_yolo(gt[tipo_caja], ancho, alto)
            if res is None or res[4] < min_lado_px or res[5] < min_lado_px:
                stats["cajas_pequenas"] += 1
                continue
            xc, yc, w, h = res[:4]
            lineas.append(f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")

        (dir_lbl / f"{id_img}.txt").write_text("\n".join(lineas))
        destino_img = dir_img / f"{id_img}.jpg"
        if not destino_img.exists():
            if mover:
                shutil.move(str(ruta_img), destino_img)
            else:
                shutil.copy2(ruta_img, destino_img)

        stats["imagenes"] += 1
        stats["cajas"] += len(lineas)
        if not lineas:
            stats["imagenes_sin_personas"] += 1

    return dict(stats)


def generar_yaml(destino, tipo_caja="vbox", ruta_yaml=None):
    """Crea el archivo .yaml que Ultralytics necesita para entrenar."""
    destino = Path(destino)
    ruta_yaml = Path(ruta_yaml) if ruta_yaml else destino / "crowdhuman.yaml"
    contenido = (
        f"# CrowdHuman en formato YOLO - caja: {tipo_caja} ({TIPOS_CAJA[tipo_caja]})\n"
        f"path: {destino}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: persona\n"
    )
    ruta_yaml.parent.mkdir(parents=True, exist_ok=True)
    ruta_yaml.write_text(contenido)
    return ruta_yaml


def dibujar_etiquetas(ruta_img, ruta_txt, color=(0, 255, 0), grosor=2):
    """Dibuja las cajas YOLO sobre la imagen para verificar la conversión. Regresa RGB."""
    img = cv2.imread(str(ruta_img))
    alto, ancho = img.shape[:2]
    texto = Path(ruta_txt).read_text().strip()
    n = 0
    for linea in texto.splitlines() if texto else []:
        _, xc, yc, w, h = map(float, linea.split())
        x1 = int((xc - w / 2) * ancho)
        y1 = int((yc - h / 2) * alto)
        x2 = int((xc + w / 2) * ancho)
        y2 = int((yc + h / 2) * alto)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, grosor)
        n += 1
    cv2.putText(img, f"personas: {n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def muestras_aleatorias(destino, split="val", n=4, semilla=0):
    """Regresa n pares (imagen, etiqueta) al azar para inspección visual."""
    destino = Path(destino)
    imagenes = sorted((destino / "images" / split).glob("*.jpg"))
    random.Random(semilla).shuffle(imagenes)
    return [(p, destino / "labels" / split / f"{p.stem}.txt") for p in imagenes[:n]]
