"""
procesamiento.py - Detección, segmentación y extracción de variables para la página web.

Reutiliza la calibración y la detección de grilla de analisis_harina_pescado.py y
replica la segmentación de segmentar_harina_v3.py (umbral Otsu local, erosión del
borde, recorte de percentiles 2-98 y normalización con el blanco de la bandeja),
para que la página calcule exactamente las mismas variables que el dataset de
entrenamiento (dataset_segmentado_v3.xlsx).

Requiere analisis_harina_pescado.py en la misma carpeta.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import analisis_harina_pescado as A
from extraccion import calcular_variables, extraer_caracteristicas


def leer_lista(texto):
    """'F1C1, F2C3' o '212 213 215' -> lista de elementos sin vacíos."""
    import re
    return [t for t in re.split(r"[,;\s]+", str(texto or "").strip()) if t]

BANDAS = {"Green": "_MS_G.TIF", "Red": "_MS_R.TIF", "RedEdge": "_MS_RE.TIF", "NIR": "_MS_NIR.TIF"}
B = list(BANDAS)
EROSION_PX = 25          # igual que segmentar_harina_v3.py
MARGEN = 0.42            # igual que el valor por defecto de analisis_harina_pescado.py

ESQUINAS = ["Arriba izquierda", "Arriba derecha", "Abajo izquierda", "Abajo derecha"]
RECORRIDOS = ["Por filas", "Por columnas"]


# --------------------------------------------------------------------------
# Lectura y calibración
# --------------------------------------------------------------------------
def buscar_archivos(carpeta: Path):
    """Devuelve {prefijo: {banda: ruta}} con las capturas completas de una carpeta."""
    capturas = {}
    for f in carpeta.iterdir():
        nombre = f.name.upper()
        for b, suf in BANDAS.items():
            if nombre.endswith(suf):
                capturas.setdefault(f.name[: -len(suf)], {})[b] = f
    return capturas


def senal_sin_irradiancia(tif: Path):
    """Calibración de analisis_harina_pescado.py sin dividir por la irradiancia del dron
    (la normalización la hace el blanco local), igual que mejorar_modelos_v2.py."""
    raw = np.array(Image.open(tif)).astype(np.float64)
    meta = A.read_metadata(tif)
    v = A.vignetting_correct(raw, meta)
    s = 2.0 ** meta["bits"]
    return raw, ((v / s - meta["black_level"] / s) / (meta["sensor_gain"] * meta["exposure_us"] / 1e6)) * meta["gain_adj"]


# --------------------------------------------------------------------------
# Ubicación de cada muestra
# --------------------------------------------------------------------------
def recuadro_muestra_unica(G: np.ndarray):
    """Una sola muestra por foto: el componente oscuro (harina) más grande que no toque
    el borde de la foto; si todos lo tocan, el más cercano al centro."""
    H, W = G.shape
    f = 4
    g = cv2.resize(np.log(np.clip(G, 1e-6, None)), (W // f, H // f), interpolation=cv2.INTER_AREA)
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(cv2.GaussianBlur(g8, (7, 7), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(th, 8)
    h, w = th.shape
    cand = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 0.01 * h * w:
            continue
        toca = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
        dist = np.hypot(cen[i][0] - w / 2, cen[i][1] - h / 2)
        cand.append((toca, -area if not toca else dist, i))
    if not cand:
        raise RuntimeError("No se encontró ninguna zona de harina en la foto.")
    _, _, i = sorted(cand)[0]
    x, y, bw, bh, _ = st[i]
    cx, cy = (x + bw / 2) * f, (y + bh / 2) * f
    hw, hh = bw * f * MARGEN, bh * f * MARGEN
    return {(0, 0): (int(cx - hw), int(cy - hh), int(cx + hw), int(cy + hh))}


def detectar_grilla(G: np.ndarray, filas: int, columnas: int):
    """Busca los bloques oscuros de harina (Otsu sobre el log de Green) y arma la grilla con
    las mismas funciones de analisis_harina_pescado.py (gap_cluster y build_axis_grid), que
    completan filas o columnas faltantes o cortadas por el borde."""
    H, W = G.shape
    f = 4
    g = cv2.resize(np.log(np.clip(G, 1e-6, None)), (W // f, H // f), interpolation=cv2.INTER_AREA)
    g8 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(cv2.GaussianBlur(g8, (7, 7), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, st, cen = cv2.connectedComponentsWithStats(th, 8)
    h, w = th.shape
    cand = [(st[i, 4], cen[i], st[i, 2], st[i, 3]) for i in range(1, n)
            if 0.002 * h * w < st[i, 4] < 0.25 * h * w and max(st[i, 2], st[i, 3]) / max(1, min(st[i, 2], st[i, 3])) < 3]
    if len(cand) < 2:
        raise RuntimeError("se encontraron muy pocos bloques de harina")
    areas = np.array(sorted([c[0] for c in cand], reverse=True))
    med = np.median(areas[: filas * columnas])
    cand = [c for c in cand if 0.35 * med <= c[0] <= 2.0 * med]
    cxy = np.array([c[1] for c in cand]) * f
    tam = np.array([(c[2], c[3]) for c in cand]) * f
    med_w, med_h = np.median(tam[:, 0]), np.median(tam[:, 1])
    filas_c = np.array(A.build_axis_grid(A.gap_cluster(cxy[:, 1], 0.6 * med_h), filas, H))
    cols_c = np.array(A.build_axis_grid(A.gap_cluster(cxy[:, 0], 0.6 * med_w), columnas, W))
    hw, hh = int(med_w * MARGEN), int(med_h * MARGEN)
    return {(r, c): (int(cx - hw), int(ry - hh), int(cx + hw), int(ry + hh))
            for r, ry in enumerate(filas_c) for c, cx in enumerate(cols_c)}


def etiquetar(recuadros, filas, columnas, esquina, recorrido, inicio, excluir, codigos=None):
    """Convierte las posiciones de la foto (fila 0 = arriba, columna 0 = izquierda) en la
    numeración del usuario. Si se da una lista de códigos, se asignan en ese orden (sirve cuando
    los lotes no son correlativos); si no, se numera desde 'inicio'."""
    desde_arriba = esquina.startswith("Arriba")
    desde_izq = esquina.endswith("izquierda")
    excluir = {e.strip().upper() for e in excluir if e.strip()}
    items = []
    for (r, c), caja in recuadros.items():
        f = r + 1 if desde_arriba else filas - r
        k = c + 1 if desde_izq else columnas - c
        orden = (f - 1) * columnas + k if recorrido == "Por filas" else (k - 1) * filas + f
        items.append({"Fila": f, "Columna": k, "Pocillo": f"F{f}C{k}", "orden": orden, "caja": caja})
    items.sort(key=lambda d: d["orden"])
    items = [it for it in items if it["Pocillo"] not in excluir]
    if codigos:
        if len(codigos) != len(items):
            raise ValueError(f"se escribieron {len(codigos)} códigos pero hay {len(items)} muestras")
        for it, c in zip(items, codigos):
            it["Muestra"] = c
    else:
        for k, it in enumerate(items):
            it["Muestra"] = inicio + k
    return items


# --------------------------------------------------------------------------
# Segmentación (replica segmentar_harina_v3.py)
# --------------------------------------------------------------------------
def segmentar(L, caja):
    G = L["Green"]
    H, W = G.shape
    claros = G > np.percentile(G, 60)
    rel_g = G / np.percentile(G[claros], 50)
    x0, y0, x1, y1 = caja
    xa, xb, ya, yb = max(0, x0), min(W, x1), max(0, y0), min(H, y1)
    bw, bh = x1 - x0, y1 - y0
    X0, Y0 = max(0, int(x0 - .35 * bw)), max(0, int(y0 - .35 * bh))
    X1, Y1 = min(W, int(x1 + .35 * bw)), min(H, int(y1 + .35 * bh))
    win = G[Y0:Y1, X0:X1]
    w8 = cv2.normalize(np.log(np.clip(win, 1e-6, None)), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    _, th = cv2.threshold(cv2.GaussianBlur(w8, (9, 9), 0), 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((15, 15), np.uint8))
    _, ll, _, _ = cv2.connectedComponentsWithStats(th, 8)
    sub = ll[ya - Y0:yb - Y0, xa - X0:xb - X0]
    ids, cnt = np.unique(sub[sub > 0], return_counts=True)
    if len(ids) == 0:
        return None, None
    blob = np.zeros((H, W), np.uint8)
    blob[Y0:Y1, X0:X1] = (ll == ids[np.argmax(cnt)])
    lim = np.zeros_like(blob)
    lim[max(0, int(y0 - .12 * bh)):min(H, int(y1 + .12 * bh)),
        max(0, int(x0 - .12 * bw)):min(W, int(x1 + .12 * bw))] = 1
    blob &= lim
    cs, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blob, cs, -1, 1, -1)
    x, y, w, h = cv2.boundingRect(blob)
    nucleo = cv2.erode(blob, np.ones((2 * EROSION_PX + 1,) * 2, np.uint8)).astype(bool)
    if nucleo.sum() < 500:
        return None, None
    d1 = cv2.dilate(blob, np.ones((61, 61), np.uint8))
    d2 = cv2.dilate(blob, np.ones((181, 181), np.uint8))
    anillo = (d2 > 0) & (d1 == 0) & (rel_g > 0.5)
    f = {"px_harina_usados": int(nucleo.sum()),
         "toca_borde_foto": bool(x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2)}
    f.update(extraer_caracteristicas(L, nucleo, anillo))
    return f, nucleo


def variables(df):
    """Conserva el nombre usado por la aplicación y delega al módulo común."""
    return calcular_variables(df)


# --------------------------------------------------------------------------
# Proceso completo de una captura
# --------------------------------------------------------------------------
def procesar_captura(archivos: dict, nombre: str, cfg: dict):
    """archivos: {banda: ruta}. cfg: tipo, filas, columnas, esquina, recorrido, inicio, excluir.
    Devuelve (DataFrame con una fila por muestra, imagen de control RGB, lista de avisos)."""
    avisos = []
    L, raw_g = {}, None
    for b in B:
        raw, L[b] = senal_sin_irradiancia(archivos[b])
        if b == "Green":
            raw_g = raw
    H, W = L["Green"].shape

    if cfg["tipo"] == "Una muestra":
        recuadros = recuadro_muestra_unica(L["Green"])
        filas, columnas = 1, 1
    else:
        filas, columnas = int(cfg["filas"]), int(cfg["columnas"])
        try:
            recuadros = detectar_grilla(L["Green"], filas, columnas)
        except Exception as e1:
            try:  # respaldo: detector original de analisis_harina_pescado.py
                recuadros, _, _ = A.detect_grid(raw_g, filas, columnas, MARGEN)
            except Exception as e2:
                raise RuntimeError(f"No se pudo ubicar la grilla ({e1}; {e2})")

    items = etiquetar(recuadros, filas, columnas, cfg.get("esquina", ESQUINAS[0]),
                      cfg.get("recorrido", RECORRIDOS[0]), int(cfg["inicio"]), cfg.get("excluir", []), cfg.get("codigos"))

    # imagen de control
    G = L["Green"]
    rel_g = G / np.percentile(G[G > np.percentile(G, 60)], 50)
    vis = cv2.cvtColor(np.clip(rel_g * 200, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
    grosor, letra = max(4, W // 400), max(1.2, W / 1600)

    filas_out = []
    for it in items:
        fila = {"Captura": nombre, "Muestra": it["Muestra"], "Pocillo": it["Pocillo"],
                "Fila": it["Fila"], "Columna": it["Columna"]}
        # El recuadro muestra dónde buscó la web la muestra; el contorno verde
        # posterior indica los píxeles de harina que realmente midió.
        x0, y0, x1, y1 = it["caja"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 140, 0), grosor)
        f, nucleo = segmentar(L, it["caja"])
        if f is None:
            fila.update({"seg_ok": False, "Revisar": "No se encontró harina"})
            cv2.rectangle(vis, (x0, y0), (x1, y1), (255, 0, 0), grosor)
        else:
            fila.update({"seg_ok": True, **f, "Revisar": ""})
            cs, _ = cv2.findContours(nucleo.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, (0, 220, 0), grosor)
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        txt = str(it["Muestra"])
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, letra * 1.6, grosor)
        cv2.putText(vis, txt, (cx - tw // 2, cy + th // 2), cv2.FONT_HERSHEY_SIMPLEX, letra * 1.6,
                    (255, 215, 0), grosor + 2)
        cv2.putText(vis, it["Pocillo"], (x0 + 10, y0 + int(45 * letra)), cv2.FONT_HERSHEY_SIMPLEX, letra,
                    (255, 60, 60), max(2, grosor - 1))
        filas_out.append(fila)

    df = pd.DataFrame(filas_out)
    if "px_harina_usados" in df and df["px_harina_usados"].notna().any():
        med = df["px_harina_usados"].median()
        poco = df["px_harina_usados"] < 0.4 * med
        df.loc[poco & df.seg_ok, "Revisar"] = "Poca harina detectada (plato vacío o mal ubicado)"
        borde = df.get("toca_borde_foto", pd.Series(False, index=df.index)).fillna(False).astype(bool)
        df.loc[borde & (df.Revisar == ""), "Revisar"] = "Muestra cortada por el borde de la foto"
    if df.seg_ok.any():
        ok = df.seg_ok
        df = pd.concat([df, variables(df[ok]).reindex(df.index)], axis=1)
    else:
        avisos.append(f"{nombre}: no se pudo segmentar ninguna muestra.")
    vis = cv2.resize(vis, (W // 2, H // 2), interpolation=cv2.INTER_AREA)
    return df, vis, avisos
