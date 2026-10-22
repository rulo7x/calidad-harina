"""
app_calidad.py - Página web de calidad de harina de pescado (Mavic 3M + ML)

Ejecutar en local:
    pip install -r requirements.txt
    streamlit run app_calidad.py

Modelo esperado (guardar al final del script de entrenamiento):
    import joblib
    joblib.dump({
        "features": ["G", "R", "RE", "NIR", "NDVI"],      # mismo orden que en el entrenamiento
        "modelos": {"Proteína": modelo_proteina},          # agregar Grasa, Humedad, Ceniza cuando existan
    }, "modelo_calidad.joblib")
"""

import io
import re
import zipfile
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tifffile
from openpyxl.styles import Alignment, Font, PatternFill

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
# Nombres de columna: deben coincidir con los usados al entrenar el modelo
COL = {"G": "G", "R": "R", "RE": "RE", "NIR": "NIR", "NDVI": "NDVI"}
BANDAS = ("G", "R", "RE", "NIR")

# Límites de especificación por defecto (editables en la barra lateral)
LIMITES_DEF = {
    "Proteína": ("min", 65.0),
    "Grasa": ("max", 10.0),
    "Humedad": ("max", 10.0),
    "Ceniza": ("max", 16.0),
}

INSTITUCION = "Universidad de Piura"
MODELO_REPO = Path(__file__).parent / "modelo_calidad.joblib"
PATRON_BANDA = re.compile(r"MS_(G|R|RE|NIR)\.TIFF?$", re.IGNORECASE)

VERDE = PatternFill("solid", start_color="C6EFCE")
ROJO = PatternFill("solid", start_color="FFC7CE")
GRIS = PatternFill("solid", start_color="EDEDED")
CABECERA = PatternFill("solid", start_color="1F4E78")


# ---------------------------------------------------------------------------
# Procesamiento de imágenes
# ---------------------------------------------------------------------------
def leer_banda(archivo) -> np.ndarray:
    arr = tifffile.imread(archivo).astype(np.float32)
    if arr.ndim > 2:
        arr = arr[..., 0] if arr.shape[-1] < arr.shape[0] else arr[0]
    return arr / 65535.0


def caja_roi(shape, frac):
    h, w = shape
    dh, dw = int(h * frac / 2), int(w * frac / 2)
    cy, cx = h // 2, w // 2
    return cy - dh, cy + dh, cx - dw, cx + dw


def calcular_features(bandas: dict, frac: float) -> dict:
    y0, y1, x0, x1 = caja_roi(bandas["NIR"].shape, frac)
    rois = {b: bandas[b][y0:y1, x0:x1] for b in BANDAS}
    feats = {COL[b]: float(np.nanmean(rois[b])) for b in BANDAS}
    nir, red = rois["NIR"], rois["R"]
    feats[COL["NDVI"]] = float(np.nanmean((nir - red) / np.clip(nir + red, 1e-6, None)))
    return feats


def compuesto_falso_color(bandas, frac):
    def estirar(a):
        lo, hi = np.percentile(a, (2, 98))
        return np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)

    rgb = np.dstack([estirar(bandas["NIR"]), estirar(bandas["R"]), estirar(bandas["G"])])
    y0, y1, x0, x1 = caja_roi(bandas["NIR"].shape, frac)
    t = max(2, rgb.shape[0] // 300)
    amarillo = np.array([1.0, 1.0, 0.0])
    rgb[y0:y0 + t, x0:x1] = amarillo
    rgb[y1 - t:y1, x0:x1] = amarillo
    rgb[y0:y1, x0:x0 + t] = amarillo
    rgb[y0:y1, x1 - t:x1] = amarillo
    return rgb


def orden_numerico(nombre: str):
    m = re.search(r"\d+", nombre)
    return (int(m.group()) if m else float("inf"), nombre)


def procesar_zip(archivo_zip, frac, barra):
    """Agrupa los TIFF por carpeta (muestra) y por captura; promedia si hay varias capturas."""
    filas, avisos = [], []
    with zipfile.ZipFile(archivo_zip) as z:
        grupos = defaultdict(lambda: defaultdict(dict))
        for info in z.infolist():
            if info.is_dir():
                continue
            p = PurePosixPath(info.filename)
            if "__MACOSX" in p.parts or p.name.startswith("._"):
                continue
            m = PATRON_BANDA.search(p.name)
            if not m:
                continue
            muestra = p.parent.name or "(sin carpeta)"
            captura = p.name[:m.start()]
            grupos[muestra][captura][m.group(1).upper()] = info.filename

        muestras = sorted(grupos, key=orden_numerico)
        for i, muestra in enumerate(muestras, start=1):
            barra.progress(i / len(muestras), text=f"Procesando muestra {muestra} ({i} de {len(muestras)})")
            lista = []
            for captura, archivos in grupos[muestra].items():
                faltan = [b for b in BANDAS if b not in archivos]
                if faltan:
                    avisos.append(f"Muestra {muestra}: captura {captura or '(sin nombre)'} incompleta, "
                                  f"faltan {', '.join(faltan)}. Se omitió.")
                    continue
                bandas = {b: leer_banda(io.BytesIO(z.read(archivos[b]))) for b in BANDAS}
                lista.append(calcular_features(bandas, frac))
            if not lista:
                avisos.append(f"Muestra {muestra}: sin capturas completas. Se omitió.")
                continue
            fila = {"Muestra": muestra, "Capturas": len(lista)}
            fila.update(pd.DataFrame(lista).mean().to_dict())
            filas.append(fila)
    return pd.DataFrame(filas), avisos


# ---------------------------------------------------------------------------
# Modelo y evaluación
# ---------------------------------------------------------------------------
def cargar_modelo(archivo):
    b = joblib.load(archivo)
    if isinstance(b, dict) and "modelos" in b:
        return b["modelos"], list(b["features"])
    feats = list(getattr(b, "feature_names_in_", [COL[k] for k in (*BANDAS, "NDVI")]))
    return {"Proteína": b}, feats


def cumple(valor, regla):
    tipo, lim = regla
    return valor >= lim if tipo == "min" else valor <= lim


def evaluar(df, modelos, features, limites):
    """Agrega predicciones, columnas de cumplimiento y la calidad general."""
    res = df.copy()
    if not modelos:
        res["Calidad"] = "Sin modelo"
        return res, []
    general = pd.Series(True, index=res.index)
    params = []
    for p, mdl in modelos.items():
        res[f"{p} (%)"] = np.round(mdl.predict(res[features]), 2)
        params.append(p)
        if p in limites:
            ok = res[f"{p} (%)"].apply(lambda v, r=limites[p]: cumple(v, r))
            res[f"{p} cumple"] = np.where(ok, "Sí", "No")
            general &= ok
    res["Calidad"] = np.where(general, "Conforme", "No conforme")
    return res, params


def generar_excel(res, info) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        res.to_excel(w, sheet_name="Resultados", index=False)
        pd.DataFrame(list(info.items()), columns=["Campo", "Valor"]).to_excel(
            w, sheet_name="Información", index=False)

        for nombre, ws in w.sheets.items():
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = CABECERA
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for col in ws.columns:
                largo = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(12, largo + 2), 50)
            ws.freeze_panes = "A2"

        ws = w.sheets["Resultados"]
        cab = [c.value for c in ws[1]]
        for j, nombre in enumerate(cab, start=1):
            for fila in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                c = fila[0]
                if nombre in (*[COL[b] for b in BANDAS], COL["NDVI"]):
                    c.number_format = "0.0000"
                elif nombre.endswith("(%)"):
                    c.number_format = "0.00"
                if nombre.endswith("cumple") or nombre == "Calidad":
                    c.alignment = Alignment(horizontal="center")
                    c.fill = {"Sí": VERDE, "Conforme": VERDE, "No": ROJO,
                              "No conforme": ROJO}.get(c.value, GRIS)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Interfaz
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Calidad harina de pescado", layout="wide")
st.title("Control de calidad de harina de pescado")
st.caption(f"Predicción de parámetros fisicoquímicos a partir de imágenes multiespectrales DJI Mavic 3M · {INSTITUCION}")

st.markdown(
    "Esta página estima la **calidad de lotes de harina de pescado** (proteína, grasa, humedad y ceniza) "
    "a partir de capturas multiespectrales tomadas con dron, sin esperar el análisis de laboratorio."
)

# El modelo solo se carga desde el repositorio (modelo_calidad.joblib); los visitantes no pueden cambiarlo
modelos, features = {}, []
if MODELO_REPO.exists():
    try:
        modelos, features = cargar_modelo(MODELO_REPO)
    except Exception as e:
        st.error(f"No se pudo cargar el modelo del repositorio: {e}")

with st.sidebar:
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

# --- Guía ------------------------------------------------------------------
g1, g2 = st.columns(2)
with g1:
    st.subheader("Cómo subir sus muestras")
    st.warning(
        "Cada foto debe contener **una sola muestra, centrada**. La página mide solo el recuadro central "
        "de la imagen, así que si una foto tiene varias muestras en grilla, las mezclará en un único resultado."
    )
    st.markdown(
        "1. Cree **una carpeta por muestra**, nombrada con su número (1, 2, 3...).\n"
        "2. Dentro de cada carpeta copie las **4 bandas TIFF** de la captura: "
        "`_MS_G`, `_MS_R`, `_MS_RE` y `_MS_NIR`.\n"
        "3. Seleccione todas las carpetas y comprímalas en un **ZIP** "
        "(en Windows: clic derecho > Enviar a > Carpeta comprimida).\n"
        "4. Suba el ZIP abajo y espere a que termine el procesamiento."
    )
    st.code(
        "muestras.zip\n"
        "├── 1/\n"
        "│   ├── DJI_0010_MS_G.TIF\n"
        "│   ├── DJI_0010_MS_R.TIF\n"
        "│   ├── DJI_0010_MS_RE.TIF\n"
        "│   └── DJI_0010_MS_NIR.TIF\n"
        "├── 2/\n"
        "│   └── ...\n"
        "└── 3/\n"
        "    └── ...",
        language=None,
    )
with g2:
    st.subheader("Qué va a obtener")
    st.markdown(
        "Un **archivo Excel** con una fila por muestra que incluye:\n"
        "- La **reflectancia media** de cada banda (Verde, Rojo, Red Edge, NIR) en la zona central de la imagen.\n"
        "- El índice **NDVI**.\n"
        "- Los **valores estimados** de cada parámetro de calidad.\n"
        "- Si cada parámetro **cumple** con la especificación, en verde o rojo.\n"
        "- La **calidad general** de la muestra: *Conforme* o *No conforme*.\n\n"
        "Una segunda hoja registra la configuración usada (zona de medición, límites y fecha). "
        "Si una carpeta tiene varias capturas, se promedian. "
        "También puede subir las 4 bandas sueltas de una sola muestra para ver su imagen."
    )
    st.caption("Las capturas completas del Mavic 3M pesan bastante: si el ZIP es muy grande, divídalo en varios.")

st.divider()

subidos = st.file_uploader(
    "Subir un ZIP con las carpetas de muestras, o las 4 bandas TIFF de una sola muestra",
    type=["zip", "tif", "tiff"], accept_multiple_files=True,
)

if subidos:
    zips = [f for f in subidos if f.name.lower().endswith(".zip")]
    tiffs = [f for f in subidos if not f.name.lower().endswith(".zip")]

    # --- Lote desde ZIP ----------------------------------------------------
    if zips:
        tablas, avisos = [], []
        barra = st.progress(0.0, text="Leyendo ZIP...")
        for z in zips:
            try:
                t, a = procesar_zip(z, frac, barra)
                tablas.append(t)
                avisos += a
            except zipfile.BadZipFile:
                avisos.append(f"{z.name}: no es un ZIP válido.")
        barra.empty()
        df = pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()

        for a in avisos:
            st.warning(a)

        if df.empty:
            st.error("No se encontraron muestras con las 4 bandas completas. Revise la estructura del ZIP.")
        else:
            res, params = evaluar(df, modelos, features, limites)

            k1, k2, k3 = st.columns(3)
            k1.metric("Muestras procesadas", len(res))
            if params:
                n_ok = int((res["Calidad"] == "Conforme").sum())
                k2.metric("Conformes", n_ok)
                k3.metric("No conformes", len(res) - n_ok)

            def colorear(v):
                if v in ("Sí", "Conforme"):
                    return "background-color: #C6EFCE; color: #006100"
                if v in ("No", "No conforme"):
                    return "background-color: #FFC7CE; color: #9C0006"
                return ""

            cols_estado = [c for c in res if c.endswith("cumple") or c == "Calidad"]
            st.dataframe(res.style.map(colorear, subset=cols_estado).format(
                {c: "{:.4f}" for c in (*[COL[b] for b in BANDAS], COL["NDVI"])}),
                use_container_width=True, hide_index=True)

            info = {
                "Fecha de procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "Archivos ZIP": ", ".join(z.name for z in zips),
                "Muestras procesadas": len(res),
                "Zona de medición (ROI central)": f"{frac:.0%} de la imagen",
                "Modelo": ", ".join(modelos) if modelos else "Sin modelo (solo datos de bandas)",
            }
            for p, (tipo, lim) in limites.items():
                info[f"Límite {p}"] = f"{'≥' if tipo == 'min' else '≤'} {lim} %"
            info["Nota"] = "Valores estimados; no reemplazan el análisis de laboratorio certificado."

            st.download_button(
                "Descargar resultados en Excel",
                generar_excel(res, info),
                f"calidad_harina_{datetime.now():%Y%m%d_%H%M}.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

    # --- Muestra individual desde TIFF -------------------------------------
    elif tiffs:
        bandas = {}
        for f in tiffs:
            m = PATRON_BANDA.search(f.name)
            if m:
                bandas[m.group(1).upper()] = leer_banda(f)
        faltan = [b for b in BANDAS if b not in bandas]
        if faltan:
            st.warning(f"Faltan bandas: {', '.join(faltan)}")
        else:
            feats = calcular_features(bandas, frac)
            c1, c2 = st.columns([3, 2])
            with c1:
                st.image(compuesto_falso_color(bandas, frac), caption="Falso color NIR-R-G (ROI en amarillo)",
                         use_container_width=True)
            with c2:
                st.subheader("Reflectancia media ROI")
                st.dataframe(pd.Series(feats, name="Valor").round(4), use_container_width=True)

            if modelos:
                res, params = evaluar(pd.DataFrame([feats]), modelos, features, limites)
                st.subheader(f"Calidad: {res['Calidad'].iloc[0]}")
                cols = st.columns(len(params))
                for col, p in zip(cols, params):
                    v = res[f"{p} (%)"].iloc[0]
                    estado = ("Cumple" if cumple(v, limites[p]) else "No cumple") if p in limites else ""
                    col.metric(p, f"{v:.2f} %", estado,
                               delta_color="normal" if estado == "Cumple" else "inverse")

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
