"""
analisis_harina_pescado.py

Procesa capturas del DJI Mavic 3M (4 bandas: Green, NIR, Red, Red Edge) de una
bandeja con pocillos (platos) de harina de pescado: calibra cada banda a un
valor de reflectancia comparable (corrige viñeteado, nivel negro, ganancia,
tiempo de exposición e irradiancia solar, usando los metadatos que trae cada
TIFF), segmenta automáticamente los pocillos en una grilla, y exporta
media/mínimo/máximo/desviación estándar por pocillo y por banda a un Excel.

------------------------------------------------------------------------
REQUISITOS
------------------------------------------------------------------------
Python 3.9+ con estas librerías (no necesitas instalar nada más, ni
exiftool ni ningún otro programa externo: los metadatos se leen con
Pillow; 'defusedxml' es necesario para que Pillow pueda leer el XMP):
       pip install numpy opencv-python-headless pillow scipy openpyxl pandas defusedxml

------------------------------------------------------------------------
USO
------------------------------------------------------------------------
Cada "captura" es una carpeta que contiene los 4 archivos de UNA sola toma:
    DJI_xxxxx_MS_G.TIF   DJI_xxxxx_MS_NIR.TIF
    DJI_xxxxx_MS_R.TIF   DJI_xxxxx_MS_RE.TIF

    python analisis_harina_pescado.py --captura "ruta/a/carpeta" ^
        --filas 4 --columnas 4 --fecha 2026-09-15 --salida resultados.xlsx

Puedes correrlo una vez por cada fecha/captura apuntando siempre al mismo
--salida: cada corrida AGREGA filas nuevas al Excel sin borrar las anteriores
(no corras dos veces la misma fecha o vas a duplicar filas).

Si el número de filas/columnas de tu bandeja no es 4x4, cambia --filas y
--columnas. Si la grilla detectada no calza bien con tus pocillos, revisa
la imagen de control que el script guarda (*_grilla.png) junto al Excel y
ajusta --margen (0.0 a 0.9, qué tan "achicado" queda cada recuadro respecto
al pocillo detectado; por defecto 0.42, o sea el recuadro final mide ~84%
del pocillo).

------------------------------------------------------------------------
UMBRALES DE DETECCION (bandeja / pocillo)
------------------------------------------------------------------------
Por defecto, el script calcula todo automáticamente y se adapta solo a
capturas con distinta luz/exposición entre fechas distintas, incluyendo
fotos con la bandeja muy de cerca (llenando casi todo el cuadro) y con
harto degrade de luz de una esquina a otra: la bandeja se separa del fondo
con un umbral que se recalcula localmente en cada zona de la foto, y cada
pocillo se separa de la superficie de la bandeja con un umbral que se
recalcula por separado en cada fila (para que una fila más oscura por el
degrade de luz no arruine la detección de esa fila).

Si en alguna captura el detector falla igual (por ejemplo "muy pocos
pocillos detectados" con luz muy pareja o muy poco contraste), puedes forzar
un valor manual con --umbral-bandeja y/o --umbral-pocillo (0-255; mientras
más alto, más estricto para considerar un pixel "claro"; esto reemplaza TODO
el calculo automatico de esa etapa por un solo valor fijo para toda la
imagen). Revisa el archivo *_grilla.png de esa captura después de ajustar,
para confirmar que la grilla quedó bien puesta.
------------------------------------------------------------------------
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import find_peaks

BAND_SUFFIXES = {
    "Green": "_MS_G.TIF",
    "NIR": "_MS_NIR.TIF",
    "Red": "_MS_R.TIF",
    "RedEdge": "_MS_RE.TIF",
}


# ------------------------------------------------------------------ #
# 1. Metadatos y calibración radiométrica (según la guía oficial DJI
#    "Mavic 3M Image Processing Guide" y su fórmula de reflectancia)
# ------------------------------------------------------------------ #
def read_metadata(tif_path: Path) -> dict:
    """
    Lee los metadatos XMP de calibracion de un TIFF del Mavic 3M usando
    unicamente Pillow (no requiere instalar exiftool ni ningun programa
    externo).
    """
    im = Image.open(tif_path)
    xmp = im.getxmp()
    try:
        desc = xmp["xmpmeta"]["RDF"]["Description"]
    except KeyError:
        raise RuntimeError(f"{tif_path.name} no tiene metadatos XMP del Mavic 3M (¿es el archivo correcto?)")

    def scalar(key):
        """Los tags DJI a veces se repiten en 2 namespaces; PIL los junta en
        una lista con el mismo valor repetido -> nos quedamos con el primero."""
        v = desc[key]
        if isinstance(v, list):
            return v[0]
        return v

    def seq(key):
        """Campos tipo VignettingCenter/VignettingPolynomial vienen como
        {'Seq': {'li': ['v1', 'v2', ...]}}"""
        v = desc[key]
        if isinstance(v, list):
            v = v[0]
        items = v["Seq"]["li"]
        if isinstance(items, str):
            items = [items]
        return [float(x) for x in items]

    bits = im.tag_v2.get(258)  # BitsPerSample (tupla, ej. (16,))
    bits = float(bits[0]) if bits else 16.0

    return {
        "bits": bits,
        "black_level": float(scalar("BlackLevel")),
        "sensor_gain": float(scalar("SensorGain")),
        "gain_adj": float(scalar("SensorGainAdjustment")),
        "exposure_us": float(scalar("ExposureTime")),
        "irradiance": float(scalar("Irradiance")),
        "vign_poly": seq("VignettingPolynomial") if "VignettingPolynomial" in desc else seq("VignettingData"),
        "vign_center": seq("VignettingCenter"),
        "band_name": scalar("BandName"),
    }


def vignetting_correct(raw: np.ndarray, meta: dict) -> np.ndarray:
    """Corrige la caida de luz hacia los bordes (viñeteado) de la lente."""
    h, w = raw.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = meta["vign_center"]
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gain = np.ones_like(r)
    rp = r.copy()
    for k in meta["vign_poly"]:
        gain += k * rp
        rp *= r
    return raw.astype(np.float64) * gain


def to_reflectance(raw_vignetted: np.ndarray, meta: dict) -> np.ndarray:
    """
    Convierte el valor crudo (ya corregido de vineteado) a un valor de
    reflectancia NORMALIZADO Y COMPARABLE entre imagenes/bandas/fechas
    (compensa nivel negro, ganancia, tiempo de exposicion e irradiancia
    solar). No es reflectancia ABSOLUTA calibrada con panel de referencia
    (para eso necesitarias fotografiar un panel de reflectancia conocida
    en cada vuelo y usarlo como factor de escala adicional), pero para
    COMPARAR platos entre si y a lo largo del tiempo, este valor ya es
    directamente comparable.
    """
    scale = 2.0 ** meta["bits"]
    I_x = raw_vignetted / scale
    I_black = meta["black_level"] / scale
    x_camera = (I_x - I_black) / (meta["sensor_gain"] * (meta["exposure_us"] / 1e6))
    x_ref = x_camera * meta["gain_adj"] / meta["irradiance"]
    return x_ref


def load_band_reflectance(tif_path: Path):
    raw = np.array(Image.open(tif_path)).astype(np.float64)
    meta = read_metadata(tif_path)
    corrected = vignetting_correct(raw, meta)
    reflect = to_reflectance(corrected, meta)
    return raw, reflect, meta


# ------------------------------------------------------------------ #
# 2. Deteccion automatica de la grilla de pocillos
# ------------------------------------------------------------------ #
def valley_threshold(values: np.ndarray, smooth: int = 9, prom_frac: float = 0.02, distance: int = 8) -> float:
    """
    Busca el punto de separacion entre 2 grupos de una distribucion con dos
    modas (dos "montañas" en el histograma): ubica los 2 picos MAS A LA
    IZQUIERDA del histograma suavizado (el pico oscuro y el pico claro,
    aunque uno de los dos sea mucho mas chico que el otro) y devuelve el
    punto mas bajo (valle) entre ambos.

    Se ignoran a proposito los valores 0 y 255 puros del histograma: son un
    artefacto de la normalizacion de contraste (percentil 1-99), no una
    moda real, y si no se ignoraran podrian generar un "pico falso" justo
    al lado del pico oscuro real y arruinar la busqueda del valle.

    Usar los 2 picos MAS A LA IZQUIERDA (en vez de los 2 MAS ALTOS, que es
    lo que hace Otsu) evita que un brillo puntual (un reflejo o brillo de
    la bandeja, que a veces genera un pico angosto pero muy alto cerca del
    255) se confunda con la moda clara real.
    """
    values = values.reshape(-1)
    hist, _ = np.histogram(values, bins=256, range=(0, 256))
    hist = hist.astype(np.float64)
    hist[0:2] = 0.0
    hist[254:256] = 0.0
    kernel = np.ones(smooth) / smooth
    hist_smooth = np.convolve(hist, kernel, mode="same")

    peaks, _ = find_peaks(hist_smooth, prominence=hist_smooth.max() * prom_frac, distance=distance)
    if len(peaks) < 2:
        lo_v, hi_v = int(np.percentile(values, 5)), int(np.percentile(values, 95))
        if hi_v <= lo_v + 1:
            return float((lo_v + hi_v) / 2)
        return float(lo_v + int(np.argmin(hist_smooth[lo_v:hi_v + 1])))

    p1, p2 = peaks[0], peaks[1]
    valley = p1 + int(np.argmin(hist_smooth[p1:p2 + 1]))
    return float(valley)


def gap_cluster(values: np.ndarray, gap_thresh: float):
    """
    Agrupa una lista de coordenadas (todos los centros X o todos los
    centros Y de los pocillos detectados) en "filas" o "columnas" reales,
    juntando valores que estan a menos de gap_thresh de distancia entre
    si. A diferencia de kmeans (que SIEMPRE arma exactamente K grupos,
    aunque los datos no den para tantos), esto arma tantos grupos como
    real y naturalmente haya en los datos: si una fila entera de la
    bandeja esta vacia (sin nada de harina, entonces sin ningun pocillo
    detectado ahi), simplemente no aparece ningun grupo en esa posicion,
    en vez de deformar los demas grupos para rellenar el hueco.
    """
    values = np.sort(values)
    groups = [[values[0]]]
    for v in values[1:]:
        if v - groups[-1][-1] > gap_thresh:
            groups.append([v])
        else:
            groups[-1].append(v)
    return [float(np.mean(g)) for g in groups]


def build_axis_grid(group_centers, n_total: int, extent: float):
    """
    A partir de los grupos de pocillos REALMENTE detectados en un eje
    (filas o columnas), arma la lista completa de n_total posiciones
    regularmente espaciadas.

    Si ya se detectaron n_total grupos, se devuelven tal cual. Si faltan
    grupos (por ejemplo, una fila entera de la bandeja vino vacia y no
    genero ningun pocillo detectado ahi), se calcula el espaciado (pitch)
    entre los grupos que SI se detectaron, y se completan los que faltan
    extrapolando ese espaciado hacia arriba y/o abajo (o izquierda/
    derecha) de la bandeja, decidiendo de que lado van los que faltan
    segun cuanto margen libre quede a cada lado en la imagen (el lado
    con espacio de sobra, del tamaño de un pocillo, es el que tenia la
    fila/columna vacia).
    """
    group_centers = sorted(group_centers)
    if len(group_centers) >= n_total:
        return group_centers[:n_total]

    diffs = np.diff(group_centers)
    pitch = float(np.min(diffs)) if len(diffs) else extent / n_total
    n_missing = n_total - len(group_centers)

    margin_top = group_centers[0] - pitch / 2
    margin_bottom = extent - (group_centers[-1] + pitch / 2)
    n_top = max(0, round(margin_top / pitch))
    n_bottom = max(0, round(margin_bottom / pitch))
    total_est = n_top + n_bottom
    if total_est == 0:
        n_top = n_missing
    elif total_est != n_missing:
        n_top = round(n_missing * n_top / total_est)
        n_bottom = n_missing - n_top

    result = list(group_centers)
    for i in range(1, n_top + 1):
        result.append(group_centers[0] - i * pitch)
    for i in range(1, n_bottom + 1):
        result.append(group_centers[-1] + i * pitch)
    return sorted(result)


def detect_grid(raw: np.ndarray, nrows: int, ncols: int, margin: float = 0.42,
                 invertir_filas: bool = False, invertir_columnas: bool = False,
                 umbral_bandeja: float = None, umbral_pocillo: float = None):
    """
    Detecta la bandeja (region clara) y los pocillos (blobs oscuros dentro
    de ella), ajusta una grilla regular nrows x ncols (para que funcione
    aunque algun pocillo individual no se detecte bien, o incluso aunque
    una FILA O COLUMNA ENTERA venga vacia de harina), y devuelve un
    diccionario {(fila, columna): (x0, y0, x1, y1)} con el recuadro de
    cada pocillo.

    Etapa 1 (bandeja vs. fondo): umbral ADAPTATIVO (se recalcula
    localmente en cada zona de la imagen) + CASCO CONVEXO del contorno mas
    grande, para que funcione con la bandeja fotografiada muy de cerca y
    con fuerte degrade de luz. No hace falta que quede perfecta: solo se
    usa para descartar el fondo/las esquinas oscuras antes de la etapa 2.

    Etapa 2 (pocillo vs. superficie): en vez de calcular un solo umbral
    para toda la bandeja (o incluso uno por fila, asumiendo que todas las
    filas existen), se recorre la imagen en franjas horizontales FINAS
    (mas finas que una fila) y se calcula el umbral por separado en cada
    una, para adaptarse a cualquier degrade de luz sin asumir nada sobre
    cuantas filas hay ni donde estan.

    Armado de la grilla: los pocillos detectados en la etapa 2 se agrupan
    en filas/columnas reales (ver gap_cluster) y, si falta alguna porque
    una fila o columna entera vino sin harina, se completa por
    extrapolacion (ver build_axis_grid) en vez de deformar las demas.

    umbral_bandeja / umbral_pocillo: si no se dan (None), todo lo anterior
    es automatico. Si se fuerzan manualmente (0-255), se usa ESE valor fijo
    para toda la imagen (sin adaptativo ni franjas) en la etapa
    correspondiente; usalo solo si la deteccion automatica falla en alguna
    captura particular.
    """
    lo, hi = np.percentile(raw, 1), np.percentile(raw, 99)
    norm = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1)
    img8 = (norm * 255).astype("uint8")
    h_img, w_img = img8.shape

    # Etapa 1: separa la bandeja (region clara) del fondo.
    if umbral_bandeja is not None:
        ub_msg = f"{umbral_bandeja:.0f} (manual)"
        _, bright_mask = cv2.threshold(img8, umbral_bandeja, 255, cv2.THRESH_BINARY)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    else:
        block = int(min(h_img, w_img) / 4.85)
        if block % 2 == 0:
            block += 1
        block = max(51, block)
        ub_msg = f"adaptativo (bloque={block})"
        bright_mask = cv2.adaptiveThreshold(
            img8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -10
        )
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError(
            "No se detecto la bandeja (region clara) en la imagen. "
            f"(umbral de bandeja usado: {ub_msg} - prueba forzando --umbral-bandeja)"
        )
    tray_cnt = max(contours, key=cv2.contourArea)
    # el casco convexo evita que pocillos muy oscuros que tocan el borde de
    # la bandeja "muerdan" el contorno y dejen ese pocillo fuera de la
    # mascara de bandeja (pasaria sobre todo en la fila mas oscura del
    # degrade de luz).
    tray_hull = cv2.convexHull(tray_cnt)
    tray_mask = np.zeros_like(img8)
    cv2.drawContours(tray_mask, [tray_hull], -1, 255, -1)

    # Etapa 2: dentro de la bandeja, separa los pocillos (mas oscuros) de la
    # superficie de la bandeja, en franjas horizontales finas.
    if umbral_pocillo is not None:
        up_msg = f"{umbral_pocillo:.0f} (manual)"
        _, wm = cv2.threshold(img8, umbral_pocillo, 255, cv2.THRESH_BINARY_INV)
        well_mask = cv2.bitwise_and(wm, tray_mask)
    else:
        n_franjas = 10
        well_mask = np.zeros_like(img8)
        franja_h = h_img / n_franjas
        umbrales_franja = []
        for i in range(n_franjas):
            y0 = int(i * franja_h)
            y1 = int((i + 1) * franja_h) if i < n_franjas - 1 else h_img
            franja_tray = tray_mask[y0:y1, :]
            franja_img = img8[y0:y1, :]
            valores = franja_img[franja_tray > 0]
            if valores.size < 50:
                continue
            t = valley_threshold(valores)
            umbrales_franja.append(t)
            franja_well = ((franja_img < t) & (franja_tray > 0)).astype(np.uint8) * 255
            well_mask[y0:y1, :] = franja_well
        up_msg = "automatico por franja (" + ", ".join(f"{v:.0f}" for v in umbrales_franja) + ")"

    well_mask = cv2.morphologyEx(well_mask, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(well_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    if not areas:
        raise RuntimeError(
            "No se detecto ningun pocillo dentro de la bandeja. "
            f"(umbral de pocillo usado: {up_msg} - prueba forzando --umbral-pocillo)"
        )
    area_thresh = max(areas) * 0.35
    good = []
    for c in contours:
        if cv2.contourArea(c) <= area_thresh:
            continue
        x, y, w, h = cv2.boundingRect(c)
        # descarta franjas/manchas alargadas que no tienen forma de pocillo
        # (por ejemplo una sombra angosta que colo por el filtro de area)
        if max(w, h) / max(1, min(w, h)) > 3.0:
            continue
        good.append(c)

    centroids, sizes = [], []
    for c in good:
        M = cv2.moments(c)
        centroids.append((M["m10"] / M["m00"], M["m01"] / M["m00"]))
        x, y, w, h = cv2.boundingRect(c)
        sizes.append((w, h))
    centroids = np.array(centroids)
    sizes = np.array(sizes)

    if len(centroids) < 2:
        raise RuntimeError(
            f"Solo se detectaron {len(centroids)} pocillos, muy pocos para "
            f"ajustar una grilla de {nrows}x{ncols}. Umbrales usados: "
            f"bandeja={ub_msg}, pocillo={up_msg}. Revisa el contraste "
            f"de la imagen, o fuerza otros valores con --umbral-bandeja y "
            f"--umbral-pocillo."
        )

    med_w, med_h = np.median(sizes[:, 0]), np.median(sizes[:, 1])

    row_groups = gap_cluster(centroids[:, 1], 0.6 * med_h)
    col_groups = gap_cluster(centroids[:, 0], 0.6 * med_w)
    row_centers = np.array(build_axis_grid(row_groups, nrows, h_img))
    col_centers = np.array(build_axis_grid(col_groups, ncols, w_img))

    # F1C1 es por defecto el pocillo arriba-a-la-izquierda de la FOTO (no
    # necesariamente de la mesa). Si en tu mesa el orden real va al reves,
    # estas dos banderas invierten el sentido en que se numeran filas y/o
    # columnas, sin tocar la deteccion en si.
    if invertir_filas:
        row_centers = row_centers[::-1]
    if invertir_columnas:
        col_centers = col_centers[::-1]

    hw, hh = int(med_w * margin), int(med_h * margin)
    boxes = {}
    for r, ry in enumerate(row_centers):
        for c, cx in enumerate(col_centers):
            boxes[(r, c)] = (int(cx - hw), int(ry - hh), int(cx + hw), int(ry + hh))
    return boxes, img8, tray_cnt


def parse_pocillo_label(label: str):
    """Convierte una etiqueta tipo 'F1C1' a (fila, columna) en base 0."""
    label = label.strip().upper()
    if not label.startswith("F") or "C" not in label:
        raise ValueError(f"'{label}' no tiene el formato esperado, ej: F1C1")
    f_part, c_part = label[1:].split("C")
    return int(f_part) - 1, int(c_part) - 1


def save_grid_preview(img8, boxes, out_path, excluidos=None):
    excluidos = excluidos or set()
    vis = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
    for (r, c), (x0, y0, x1, y1) in boxes.items():
        if (r, c) in excluidos:
            # celda sin pocillo fisico (ej. una esquina vacia de la bandeja):
            # se dibuja en gris y no se procesa ni se numera.
            cv2.rectangle(vis, (x0, y0), (x1, y1), (128, 128, 128), 4)
            cv2.putText(vis, f"F{r+1}C{c+1} (excluido)", (x0 + 8, y0 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 3)
        else:
            cv2.rectangle(vis, (x0, y0), (x1, y1), (0, 255, 0), 4)
            cv2.putText(vis, f"F{r+1}C{c+1}", (x0 + 8, y0 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
    Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).save(out_path)


# ------------------------------------------------------------------ #
# 3. Estadisticas por pocillo y por banda
# ------------------------------------------------------------------ #
def well_stats(reflect: np.ndarray, box):
    x0, y0, x1, y1 = box
    h, w = reflect.shape
    # recorte correcto a los limites de la imagen: antes, si una caja quedaba
    # TOTALMENTE fuera del cuadro (por ejemplo arriba, con y0 y y1 negativos),
    # min(h, y1) con y1 negativo no lo corregia, y el slicing con indice
    # negativo de numpy devolvia una region gigante y equivocada (casi toda
    # la imagen) en vez de dar error o region vacia. Ahora se recorta a
    # [0, ancho/alto] en ambos extremos, y si no queda area real se avisa
    # con un error claro en vez de inventar un numero.
    x0 = min(max(0, x0), w)
    x1 = min(max(0, x1), w)
    y0 = min(max(0, y0), h)
    y1 = min(max(0, y1), h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            f"La caja {box} queda totalmente fuera de la imagen (tamano {w}x{h}); "
            "no hay pixeles reales que medir ahi."
        )
    region = reflect[y0:y1, x0:x1]
    return {
        "media": float(np.mean(region)),
        "min": float(np.min(region)),
        "max": float(np.max(region)),
        "desv_std": float(np.std(region)),
    }


# ------------------------------------------------------------------ #
# 4. Procesamiento de una captura completa (4 bandas)
# ------------------------------------------------------------------ #
def procesar_captura(carpeta: Path, nrows: int, ncols: int, fecha: str, margin: float,
                      invertir_filas: bool = False, invertir_columnas: bool = False,
                      id_inicial: int = None, excluir: set = None,
                      umbral_bandeja: float = None, umbral_pocillo: float = None):
    # encuentra el prefijo comun (DJI_xxxxxxxxxxxxxx_xxxx) a partir del archivo _MS_G.TIF
    g_files_raw = list(carpeta.glob("*_MS_G.TIF")) + list(carpeta.glob("*_MS_G.tif"))
    # En Windows el sistema de archivos no distingue mayusculas/minusculas, asi
    # que las dos busquedas de arriba pueden encontrar el mismo archivo dos
    # veces. Quitamos duplicados (comparando el nombre en minusculas) para no
    # procesar la misma captura repetida.
    vistos = set()
    g_files = []
    for f in g_files_raw:
        clave = f.name.lower()
        if clave not in vistos:
            vistos.add(clave)
            g_files.append(f)
    if not g_files:
        raise RuntimeError(f"No se encontro ningun *_MS_G.TIF en {carpeta}")

    excluir = excluir or set()

    filas = []
    preview_saved = False
    for g_file in g_files:
        prefix = g_file.name.replace("_MS_G.TIF", "").replace("_MS_G.tif", "")
        band_data = {}
        boxes = None
        img8_for_preview = None
        for band_name, suffix in BAND_SUFFIXES.items():
            tif_path = carpeta / f"{prefix}{suffix}"
            if not tif_path.exists():
                raise RuntimeError(f"Falta el archivo {tif_path.name} para la captura {prefix}")
            raw, reflect, meta = load_band_reflectance(tif_path)
            band_data[band_name] = reflect
            if band_name == "Green":
                boxes, img8_for_preview, _ = detect_grid(
                    raw, nrows, ncols, margin, invertir_filas, invertir_columnas,
                    umbral_bandeja, umbral_pocillo,
                )

        preview_path = carpeta / f"{prefix}_grilla.png"
        save_grid_preview(img8_for_preview, boxes, preview_path, excluidos=excluir)
        preview_saved = True
        print(f"  -> vista previa de la grilla guardada en: {preview_path}")
        if excluir:
            etiquetas = ", ".join(f"F{r+1}C{c+1}" for (r, c) in sorted(excluir))
            print(f"  -> excluyendo (no se procesan ni se numeran): {etiquetas}")

        idx_id = 0  # contador aparte para Codigo_muestra, que solo avanza en pocillos reales
        for (r, c), box in boxes.items():
            if (r, c) in excluir:
                continue
            fila = {
                "Fecha": fecha,
                "Captura": prefix,
                "Pocillo": f"F{r+1}C{c+1}",
                "Fila": r + 1,
                "Columna": c + 1,
            }
            # boxes ya viene en orden de lectura (fila por fila, de izquierda
            # a derecha, respetando --invertir-filas/--invertir-columnas), asi
            # que solo hace falta ir sumando 1 al id inicial que nos dieron
            # (saltando los pocillos excluidos, que no cuentan).
            if id_inicial is not None:
                fila["Codigo_muestra"] = id_inicial + idx_id
            idx_id += 1
            band_means = {}
            for band_name in BAND_SUFFIXES:
                stats = well_stats(band_data[band_name], box)
                band_means[band_name] = stats["media"]
                for stat_name, val in stats.items():
                    fila[f"{band_name}_{stat_name}"] = val

            # indices opcionales combinando bandas (ajusta/quita segun lo
            # que realmente sea relevante para tu analisis de harina de
            # pescado; se incluyen solo como referencia general)
            g, nir, red, re = (band_means["Green"], band_means["NIR"],
                                band_means["Red"], band_means["RedEdge"])
            fila["NDVI"] = (nir - red) / (nir + red) if (nir + red) != 0 else None
            fila["NDRE"] = (nir - re) / (nir + re) if (nir + re) != 0 else None
            fila["GNDVI"] = (nir - g) / (nir + g) if (nir + g) != 0 else None
            filas.append(fila)

    return pd.DataFrame(filas)


def ordenar_resultados(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ordena el Excel de resultados de la primera prueba a la ultima, sin
    importar en que orden hayas ido procesando las carpetas de captura.

    Se ordena por 'Codigo_muestra' (el numero correlativo que le pusiste
    con --id-inicial) cuando esta presente, porque es el identificador real
    de "primera prueba a la ultima". Las filas que no tengan Codigo_muestra
    (por ejemplo si corriste el script alguna vez sin --id-inicial) quedan
    al final, ordenadas por Fecha/Captura/Fila/Columna para que al menos
    queden agrupadas de forma consistente.
    """
    if "Codigo_muestra" in df.columns and df["Codigo_muestra"].notna().any():
        tiene_codigo = df["Codigo_muestra"].notna()
        con_codigo = df[tiene_codigo].sort_values("Codigo_muestra", kind="stable")
        sin_codigo = df[~tiene_codigo].sort_values(
            ["Fecha", "Captura", "Fila", "Columna"], kind="stable"
        )
        return pd.concat([con_codigo, sin_codigo], ignore_index=True)
    return df.sort_values(["Fecha", "Captura", "Fila", "Columna"], kind="stable").reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captura", required=True, help="Carpeta con los 4 TIFF de una captura")
    ap.add_argument("--filas", type=int, default=4, help="Numero de filas de pocillos en la bandeja")
    ap.add_argument("--columnas", type=int, default=4, help="Numero de columnas de pocillos en la bandeja")
    ap.add_argument("--fecha", required=True, help="Fecha/etiqueta de esta captura, ej: 2026-09-15")
    ap.add_argument("--salida", default="resultados.xlsx", help="Archivo Excel de salida (se agregan filas si ya existe)")
    ap.add_argument("--margen", type=float, default=0.42, help="Que tan grande es el recuadro de lectura dentro de cada pocillo (0-0.9)")
    ap.add_argument("--invertir-filas", action="store_true",
                     help="Invierte el orden de las filas (F1 pasa a ser la fila que antes era la ultima)")
    ap.add_argument("--invertir-columnas", action="store_true",
                     help="Invierte el orden de las columnas (C1 pasa a ser la columna que antes era la ultima)")
    ap.add_argument("--id-inicial", type=int, default=None,
                     help="Si lo das, agrega una columna 'Codigo_muestra' numerando los pocillos de esta "
                          "captura de forma correlativa empezando en este numero, en el orden actual "
                          "(fila por fila, izquierda a derecha), saltando los que esten en --excluir")
    ap.add_argument("--excluir", default=None,
                     help="Lista separada por comas de celdas de la grilla que NO tienen pocillo fisico "
                          "(por ejemplo una esquina vacia de la bandeja), para que no se procesen ni se "
                          "numeren. Ejemplo: --excluir F1C1  o  --excluir F1C1,F2C3")
    ap.add_argument("--umbral-bandeja", type=float, default=None,
                     help="Fuerza el umbral (0-255) para detectar la bandeja, en vez de calcularlo "
                          "automaticamente (adaptativo). Usalo solo si la deteccion automatica falla.")
    ap.add_argument("--umbral-pocillo", type=float, default=None,
                     help="Fuerza el umbral (0-255) para detectar los pocillos dentro de la bandeja, en "
                          "vez de calcularlo automaticamente (por fila). Usalo solo si la deteccion "
                          "automatica falla.")
    args = ap.parse_args()

    excluir = set()
    if args.excluir:
        for etiqueta in args.excluir.split(","):
            excluir.add(parse_pocillo_label(etiqueta))

    carpeta = Path(args.captura)
    print(f"Procesando captura en: {carpeta}")
    df_nuevo = procesar_captura(
        carpeta, args.filas, args.columnas, args.fecha, args.margen,
        args.invertir_filas, args.invertir_columnas, args.id_inicial, excluir,
        args.umbral_bandeja, args.umbral_pocillo,
    )

    salida = Path(args.salida)
    if salida.exists():
        df_previo = pd.read_excel(salida)
        df_final = pd.concat([df_previo, df_nuevo], ignore_index=True)
    else:
        df_final = df_nuevo

    df_final = ordenar_resultados(df_final)
    df_final.to_excel(salida, index=False)
    print(f"\nListo. {len(df_nuevo)} filas nuevas agregadas a: {salida}")
    print(f"Total de filas en el archivo: {len(df_final)} (ordenado por Codigo_muestra)")


if __name__ == "__main__":
    main()
