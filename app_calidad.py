"""
app_calidad.py - Panel web de calidad de harina de pescado (Mavic 3M + ML)

Ejecutar:
    pip install streamlit tifffile joblib scikit-learn pandas numpy
    streamlit run app_calidad.py

Modelo esperado (guardar al final del script de entrenamiento):
    import joblib
    joblib.dump({
        "features": ["G", "R", "RE", "NIR", "NDVI"],      # mismo orden que en el entrenamiento
        "modelos": {"Proteína": modelo_proteina},          # agregar Grasa, Humedad, Ceniza cuando existan
    }, "modelo_calidad.joblib")
"""

import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tifffile

# ---------------------------------------------------------------------------
# Configuración (ajustar a los nombres de columna que genera paso0_extraer_bandas.py)
# ---------------------------------------------------------------------------
COL = {"G": "G", "R": "R", "RE": "RE", "NIR": "NIR", "NDVI": "NDVI"}

# Límites de especificación por defecto (editables en la barra lateral)
LIMITES_DEF = {
    "Proteína": ("min", 65.0),
    "Grasa": ("max", 10.0),
    "Humedad": ("max", 10.0),
    "Ceniza": ("max", 16.0),
}

INSTITUCION = "Universidad de Piura"

MODELO_REPO =Path(__file__).parent / "modelo_calidad.joblib"

PATRON_BANDA =re.compile(r"MS_(G|R|RE|NIR)\.TIF{1,2}$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Funciones
# ---------------------------------------------------------------------------
def leer_banda(archivo) -> np.ndarray:
    arr = tifffile.imread(archivo).astype(np.float32)
    if arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] < arr.shape[0] else arr[0]
    return arr / 65535.0


def roi_central(arr: np.ndarray, frac: float):
    h, w = arr.shape
    dh, dw = int(h * frac / 2), int(w * frac / 2)
    cy, cx = h // 2, w // 2
    return arr[cy - dh:cy + dh, cx - dw:cx + dw], (cy - dh, cy + dh, cx - dw, cx + dw)


def extraer_desde_tiffs(archivos, frac):
    bandas = {}
    for f in archivos:
        m = PATRON_BANDA.search(f.name)
        if m:
            bandas[m.group(1).upper()] = leer_banda(f)
    faltan = [b for b in ("G", "R", "RE", "NIR") if b not in bandas]
    if faltan:
        return None, None, None, faltan

    feats, rois = {}, {}
    caja = None
    for b, arr in bandas.items():
        roi, caja = roi_central(arr, frac)
        rois[b] = roi
        feats[COL[b]] = float(np.nanmean(roi))
    nir, red = rois["NIR"], rois["R"]
    ndvi = (nir - red) / np.clip(nir + red, 1e-6, None)
    feats[COL["NDVI"]] = float(np.nanmean(ndvi))
    return feats, bandas, caja, []


def compuesto_falso_color(bandas, caja):
    def estirar(a):
        lo, hi = np.percentile(a, (2, 98))
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    rgb = np.dstack([estirar(bandas["NIR"]), estirar(bandas["R"]), estirar(bandas["G"])])
    y0, y1, x0, x1 = caja
    t = max(2, rgb.shape[0] // 300)
    amarillo = np.array([1.0, 1.0, 0.0])
    rgb[y0:y0 + t, x0:x1] = amarillo
    rgb[y1 - t:y1, x0:x1] = amarillo
    rgb[y0:y1, x0:x0 + t] = amarillo
    rgb[y0:y1, x1 - t:x1] = amarillo
    return rgb


def cargar_modelo(archivo):
    b = joblib.load(archivo)
    if isinstance(b, dict) and "modelos" in b:
        return b["modelos"], list(b["features"])
    feats = list(getattr(b, "feature_names_in_", [COL[k] for k in ("G", "R", "RE", "NIR", "NDVI")]))
    return {"Proteína": b}, feats


def cumple(valor, regla):
    tipo, lim = regla
    return valor >= lim if tipo == "min" else valor <= lim


def predecir(df_feats, modelos, features):
    out = pd.DataFrame(index=df_feats.index)
    for param, mdl in modelos.items():
        out[param] = mdl.predict(df_feats[features])
    return out


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Calidad harina de pescado", layout="wide")
st.title("Control de calidad de harina de pescado")
st.caption(f"Predicción de parámetros fisicoquímicos a partir de imágenes multiespectrales DJI Mavic 3M · {INSTITUCION}")

st.markdown(
    "Esta página estima la **calidad de un lote de harina de pescado** (proteína, grasa, humedad y ceniza) "
    "a partir de una sola captura multiespectral tomada con dron, sin esperar el análisis de laboratorio. "
    "Suba las 4 bandas de la captura para ver la imagen procesada y el resultado."
)

with st.sidebar:
    st.header("Modelo")
    f_modelo = st.file_uploader(
        "Archivo .joblib (opcional)", type=["joblib", "pkl"],
        help="Si el repositorio ya incluye el modelo entrenado, se carga solo. "
             "Aquí se puede probar otro modelo sin cambiar la página.",
    )
    modelos, features = ({}, [])
    if f_modelo is None and Path(MODELO_REPO).exists():
        f_modelo = MODELO_REPO  # modelo incluido en el repositorio: carga automática
    if f_modelo:
        try:
            modelos, features = cargar_modelo(f_modelo)
            st.success(f"Parámetros: {', '.join(modelos)}")
            st.caption(f"Variables: {', '.join(features)}")
        except Exception as e:
            st.error(f"No se pudo cargar el modelo: {e}")

    st.header("Captura")
    frac = st.slider(
        "Tamaño ROI central", 0.05, 0.9, 0.30, 0.05,
        help="Fracción de la imagen, al centro, que se usa para medir la muestra. "
             "Se evita el borde para no incluir el envase ni el fondo.",
    )

    st.header("Especificación")
    st.caption("Límites contra los que se compara cada predicción para indicar si cumple. "
               "Ajústelos a la especificación comercial que corresponda.")
    limites = {}
    for p, (tipo, val) in LIMITES_DEF.items():
        etiqueta = f"{p} {'mínimo' if tipo == 'min' else 'máximo'} (%)"
        limites[p] = (tipo, st.number_input(etiqueta, value=val, step=0.5))

archivos = st.file_uploader(
    "Subir las 4 bandas de una captura (MS_G, MS_R, MS_RE, MS_NIR)",
    type=["tif", "tiff"], accept_multiple_files=True,
)
if archivos:
    feats, bandas, caja, faltan = extraer_desde_tiffs(archivos, frac)
    if faltan:
        st.warning(f"Faltan bandas: {', '.join(faltan)}")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.image(compuesto_falso_color(bandas, caja), caption="Falso color NIR-R-G (ROI en amarillo)",
                     use_container_width=True)
        with c2:
            st.subheader("Reflectancia media ROI")
            st.dataframe(pd.Series(feats, name="Valor").round(4), use_container_width=True)

        if modelos:
            pred = predecir(pd.DataFrame([feats]), modelos, features).iloc[0]
            st.subheader("Predicción")
            cols = st.columns(len(pred))
            for col, (p, v) in zip(cols, pred.items()):
                regla = limites.get(p)
                estado = ("Cumple" if cumple(v, regla) else "No cumple") if regla else ""
                col.metric(p, f"{v:.2f} %", estado,
                           delta_color="normal" if estado == "Cumple" else "inverse")
        else:
            st.info("Cargue un modelo en la barra lateral para obtener predicciones.")

# ---------------------------------------------------------------------------
# Información del proyecto
# ---------------------------------------------------------------------------
st.divider()
st.header("¿Cómo funciona?")

p1, p2, p3, p4 = st.columns(4)
with p1:
    st.markdown("**1. Captura**")
    st.markdown(
        "Se fotografía la muestra con un dron **DJI Mavic 3 Multispectral**, que registra cuatro bandas: "
        "Verde (560 nm), Rojo (650 nm), Red Edge (730 nm) e Infrarrojo cercano (860 nm)."
    )
with p2:
    st.markdown("**2. Procesamiento**")
    st.markdown(
        "Se recorta una región central (ROI) para medir solo la harina, se convierten los valores de 16 bits "
        "a reflectancia (0 a 1) y se calcula el índice NDVI."
    )
with p3:
    st.markdown("**3. Predicción**")
    st.markdown(
        "Un modelo de *machine learning* relaciona esa firma espectral con los resultados de laboratorio "
        "y estima proteína, grasa, humedad y ceniza."
    )
with p4:
    st.markdown("**4. Resultado**")
    st.markdown(
        "Cada valor se compara con los límites de especificación y se indica si la muestra cumple o no."
    )

with st.expander("Datos y entrenamiento"):
    st.markdown(
        "- Los modelos se entrenan con **contramuestras de cerca de 300 lotes de producción**, "
        "cada una con su resultado de laboratorio certificado.\n"
        "- Se parte por un solo parámetro, la **proteína**, y se amplía a los demás a medida que crece la base de datos.\n"
        "- Se comparan modelos de complejidad creciente (regresión lineal, Random Forest, SVR, redes neuronales) "
        "y se evalúan con el coeficiente de correlación (R) y el error cuadrático medio (RMSE).\n"
        "- La referencia metodológica es una tesis que predice estos parámetros con imágenes **hiperespectrales** "
        "e inteligencia artificial."
    )

with st.expander("Alcance y limitaciones"):
    st.markdown(
        "- El Mavic 3M registra **4 bandas**, frente a las cerca de 240 de una cámara hiperespectral. "
        "Por eso la precisión esperada es menor que en los trabajos de referencia, a cambio de un equipo "
        "más accesible y portátil.\n"
        "- Los resultados son **estimaciones** y no reemplazan el análisis de laboratorio certificado.\n"
        "- El proyecto está en desarrollo: los modelos se actualizan a medida que se incorporan más datos."
    )

st.caption(f"Proyecto de investigación · {INSTITUCION}")
