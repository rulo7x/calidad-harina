"""
app_calidad.py - Página web de calidad de harina de pescado (DJI Mavic 3M)

Archivos que deben estar en el repositorio, en la misma carpeta:
    app_calidad.py               esta página
    procesamiento.py             detección de grilla, segmentación y variables
    analisis_harina_pescado.py   calibración y funciones de grilla (tu script original)
    requirements.txt
    .streamlit/config.toml       (límite de subida)
    modelo_calidad.joblib        opcional; lo genera entrenar_modelo_web.py

Ejecutar en local:
    pip install -r requirements.txt
    streamlit run app_calidad.py
"""

import io
import re
import tempfile
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from openpyxl.styles import Alignment, Font, PatternFill

import procesamiento as P

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
INSTITUCION = "Universidad de Piura"
MODELO_REPO = Path(__file__).parent / "modelo_calidad.joblib"

# Límites de especificación por defecto (los nombres deben coincidir con los del modelo)
LIMITES_DEF = {
    "Proteína": ("min", 65.0),
    "Grasa": ("max", 10.0),
    "Humedad": ("max", 10.0),
    "Ceniza": ("max", 16.0),
}
TIPOS = ["Varias muestras (grilla)", "Una muestra"]

VERDE, ROJO, AMBAR, GRIS = (PatternFill("solid", start_color=c) for c in ("C6EFCE", "FFC7CE", "FFEB9C", "EDEDED"))
CABECERA = PatternFill("solid", start_color="1F4E78")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def orden_natural(texto):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(texto))]


def guardar_subida(subidos, destino: Path):
    """Copia los archivos subidos (ZIP o TIFF sueltos) a una carpeta temporal y devuelve
    la lista de capturas: [{"Carpeta", "Captura", "archivos": {banda: ruta}}]."""
    for f in subidos:
        if f.name.lower().endswith(".zip"):
            with zipfile.ZipFile(f) as z:
                for info in z.infolist():
                    p = PurePosixPath(info.filename)
                    if info.is_dir() or "__MACOSX" in p.parts or p.name.startswith("._"):
                        continue
                    if not p.name.upper().endswith((".TIF", ".TIFF")):
                        continue
                    carpeta = p.parent.name or "(raíz)"
                    out = destino / carpeta / p.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    out.write_bytes(z.read(info))
        else:
            out = destino / "(sueltos)" / f.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(f.getvalue())

    capturas, incompletas = [], []
    for carpeta in sorted([d for d in destino.iterdir() if d.is_dir()], key=lambda d: orden_natural(d.name)):
        for pref, arch in sorted(P.buscar_archivos(carpeta).items()):
            if len(arch) == 4:
                capturas.append({"Carpeta": carpeta.name, "Captura": pref, "archivos": arch})
            else:
                faltan = [b for b in P.B if b not in arch]
                incompletas.append(f"{carpeta.name} / {pref}: faltan {', '.join(faltan)}")
    return capturas, incompletas


def tabla_descripcion(capturas, tipo, filas, columnas, esquina, recorrido, inicio):
    """Tabla editable con la descripción de cada captura, prellenada con los valores por defecto."""
    n_por = 1 if tipo == "Una muestra" else filas * columnas
    filas_t = []
    for k, c in enumerate(capturas):
        if tipo == "Una muestra":
            num = re.findall(r"\d+", c["Carpeta"])
            ini = int(num[-1]) if num else inicio + k
        else:
            ini = inicio + k * n_por
        filas_t.append({"Carpeta": c["Carpeta"], "Captura": c["Captura"], "Tipo": tipo, "Filas": filas,
                        "Columnas": columnas, "Muestra 1 en": esquina, "Recorrido": recorrido,
                        "N° inicial": ini, "Códigos (opcional)": "", "Excluir (opcional)": ""})
    return pd.DataFrame(filas_t)


def mapa_numeracion(filas, columnas, esquina, recorrido, inicio=1):
    """Matriz de ejemplo con la numeración, tal como se ve en la foto."""
    cajas = {(r, c): None for r in range(filas) for c in range(columnas)}
    items = P.etiquetar(cajas, filas, columnas, esquina, recorrido, inicio, [])
    m = np.zeros((filas, columnas), int)
    for it in items:
        r = it["Fila"] - 1 if esquina.startswith("Arriba") else filas - it["Fila"]
        c = it["Columna"] - 1 if esquina.endswith("izquierda") else columnas - it["Columna"]
        m[r, c] = it["Muestra"]
    return pd.DataFrame(m, index=[f"↓{i + 1}" for i in range(filas)], columns=[f"→{j + 1}" for j in range(columnas)])


def cargar_modelo():
    if not MODELO_REPO.exists():
        return {}, []
    b = joblib.load(MODELO_REPO)
    return b["modelos"], list(b["features"])


def cumple(valor, regla):
    tipo, lim = regla
    return valor >= lim if tipo == "min" else valor <= lim


def evaluar(df, modelos, features, limites):
    res = df.copy()
    if not modelos:
        return res, []
    faltan = [f for f in features if f not in res]
    if faltan:
        st.warning("El modelo del repositorio usa variables que la página no calcula "
                   f"({', '.join(faltan[:5])}...). Se entregan solo los datos de las bandas.")
        return res, []
    ok = res["seg_ok"].fillna(False).astype(bool) & res[features].notna().all(axis=1)
    general = pd.Series(True, index=res.index)
    motivos = pd.Series([[] for _ in range(len(res))], index=res.index)
    params = []
    for p, mdl in modelos.items():
        col = f"{p} estimada (%)"
        res[col] = np.nan
        if ok.any():
            res.loc[ok, col] = np.round(np.ravel(mdl.predict(res.loc[ok, features])), 2)
        params.append(p)
        if p in limites:
            tl, lim = limites[p]
            c_ok = res[col].apply(lambda v, r=limites[p]: bool(cumple(v, r)) if pd.notna(v) else False)
            res[f"{p} cumple"] = np.where(ok, np.where(c_ok, "Sí", "No"), "")
            general &= c_ok
            for i in res.index[ok & ~c_ok]:
                v = res.at[i, col]
                motivos[i] = motivos[i] + [f"{p} {v:.2f} % ({'debajo del mínimo' if tl == 'min' else 'sobre el máximo'} "
                                           f"de {lim:g} %)"]
    res["Calidad"] = np.where(~ok, "Sin datos", np.where(general, "Conforme", "No conforme"))
    sin_datos = res["Revisar"].fillna("").replace("", "No se pudo medir la harina")
    res["Motivo"] = np.where(~ok, sin_datos,
                             np.where(general, "Cumple todos los límites", motivos.apply("; ".join)))
    res.insert(res.columns.get_loc("Calidad") + 1, "Motivo", res.pop("Motivo"))
    return res, params


def conclusiones(res, params, limites):
    """Resumen en texto de por qué las muestras salen conformes o no conformes."""
    if not params:
        return []
    medidas = res[res["Calidad"] != "Sin datos"]
    n, n_no = len(medidas), int((medidas["Calidad"] == "No conforme").sum())
    lineas = []
    if n == 0:
        return ["No se pudo medir ninguna muestra, así que no hay veredicto de calidad."]
    if n_no == 0:
        lineas.append(f"Las {n} muestras medidas cumplen todos los límites de la especificación.")
    else:
        lineas.append(f"{n_no} de {n} muestras medidas salen **no conformes**: cada una tiene al menos un parámetro "
                      "estimado fuera de la especificación.")
    causas = {}
    for p in params:
        if p not in limites or f"{p} cumple" not in medidas:
            continue
        tl, lim = limites[p]
        fallan = medidas[medidas[f"{p} cumple"] == "No"]
        if len(fallan):
            causas[p] = len(fallan)
            lado = "por debajo del mínimo" if tl == "min" else "por encima del máximo"
            lineas.append(f"**{p}:** {len(fallan)} muestra(s) quedan {lado} de {lim:g} %. "
                          f"El valor estimado va de {medidas[f'{p} estimada (%)'].min():.2f} a "
                          f"{medidas[f'{p} estimada (%)'].max():.2f} % (promedio "
                          f"{medidas[f'{p} estimada (%)'].mean():.2f} %).")
    if causas and n_no:
        principal, cuenta = max(causas.items(), key=lambda kv: kv[1])
        if cuenta == n_no:
            sin = medidas[[c for c in medidas if c.endswith("cumple") and not c.startswith(principal)]]
            pasarian = int((sin == "Sí").all(axis=1).sum())
            lineas.append(f"La causa común es **{principal}**: todas las no conformes fallan en ese parámetro. "
                          f"Si solo se evaluaran los demás parámetros, {pasarian} de {n} cumplirían.")
        rango = {p: medidas[f"{p} estimada (%)"].max() - medidas[f"{p} estimada (%)"].min() for p in causas}
        if all(v < 1.5 for v in rango.values()):
            lineas.append("El modelo estima valores muy parecidos para todas las muestras, por lo que el veredicto "
                          "depende sobre todo de dónde cae ese promedio respecto al límite, más que de diferencias "
                          "reales entre lotes.")
    n_sd = int((res["Calidad"] == "Sin datos").sum())
    if n_sd:
        lineas.append(f"{n_sd} posición(es) quedaron **sin datos** porque no se encontró harina; revise la columna Motivo.")
    lineas.append("Las estimaciones son experimentales y no reemplazan el análisis de laboratorio certificado.")
    return lineas


def generar_excel(res, descripcion, info, lineas=None) -> bytes:
    buf = io.BytesIO()
    ocultar = ["seg_ok", "toca_borde_foto"]
    with pd.ExcelWriter(buf, engine="openpyxl") as w:
        res.drop(columns=[c for c in ocultar if c in res]).to_excel(w, sheet_name="Resultados", index=False)
        if lineas:
            pd.DataFrame({"Conclusión": lineas}).to_excel(w, sheet_name="Conclusión", index=False)
        descripcion.to_excel(w, sheet_name="Descripción de capturas", index=False)
        pd.DataFrame(list(info.items()), columns=["Campo", "Valor"]).to_excel(w, sheet_name="Información", index=False)
        for ws in w.sheets.values():
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = CABECERA
                c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            for col in ws.columns:
                largo = max(len(str(c.value)) if c.value is not None else 0 for c in col)
                ws.column_dimensions[col[0].column_letter].width = min(max(11, largo + 2), 45)
            ws.freeze_panes = "A2"
        if "Conclusión" in w.sheets:
            wc = w.sheets["Conclusión"]
            wc.column_dimensions["A"].width = 110
            for (c,) in wc.iter_rows(min_row=2):
                c.alignment = Alignment(wrap_text=True, vertical="top")
        ws = w.sheets["Resultados"]
        for j, cab in enumerate([c.value for c in ws[1]], start=1):
            for (c,) in ws.iter_rows(min_row=2, min_col=j, max_col=j):
                if cab.endswith(("_rel", "_cv")) or cab.startswith(("log_", "ND_", "cv_")):
                    c.number_format = "0.0000"
                elif cab.endswith("(%)"):
                    c.number_format = "0.00"
                if cab.endswith("cumple") or cab == "Calidad":
                    c.alignment = Alignment(horizontal="center")
                    c.fill = {"Sí": VERDE, "Conforme": VERDE, "No": ROJO, "No conforme": ROJO}.get(c.value, GRIS)
                if cab == "Revisar" and c.value:
                    c.fill = AMBAR
    return buf.getvalue()


def zip_imagenes(imagenes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for nombre, img in imagenes.items():
            ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
            z.writestr(f"{re.sub(r'[^\w.-]+', '_', nombre)}.jpg", enc.tobytes())
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Página
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Calidad harina de pescado", layout="wide")
st.title("Control de calidad de harina de pescado")
st.caption(f"Imágenes multiespectrales DJI Mavic 3M · {INSTITUCION}")
st.markdown(
    "Esta página ubica cada muestra de harina en las fotos multiespectrales del dron, **segmenta solo la harina** "
    "(sin plato, bandeja ni mesa), mide sus cuatro bandas y entrega un **Excel con una fila por muestra**."
)

modelos, features = {}, []
try:
    modelos, features = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo del repositorio: {e}")

with st.sidebar:
    st.header("Especificación")
    st.caption("Límites contra los que se compara cada estimación para indicar si cumple.")
    limites = {}
    for p, (tipo_l, val) in LIMITES_DEF.items():
        etiqueta = f"{p} {'mínimo' if tipo_l == 'min' else 'máximo'} (%)"
        limites[p] = (tipo_l, st.number_input(etiqueta, value=val, step=0.5))
    if not modelos:
        st.info("Todavía no hay modelo cargado: se entregarán solo los datos de las bandas.")

# --- Paso 1: guía y subida ------------------------------------------------------
st.header("1. Prepare y suba sus fotos")
g1, g2 = st.columns(2)
with g1:
    st.markdown(
        "Cada captura del Mavic 3M son **4 archivos TIFF** (`_MS_G`, `_MS_R`, `_MS_RE`, `_MS_NIR`). "
        "Puede subir:\n"
        "- **Una sola captura:** los 4 TIFF sueltos.\n"
        "- **Varias capturas:** un **ZIP** con una carpeta numerada por captura (1, 2, 3...) "
        "y los 4 TIFF dentro de cada una.\n\n"
        "Cada foto puede tener **una sola muestra** o **varias muestras en grilla** (por ejemplo, una bandeja "
        "de 4 × 4 platos). Eso lo indica en el paso 2."
    )
with g2:
    st.code("capturas.zip\n├── 1/\n│   ├── DJI_..._MS_G.TIF\n│   ├── DJI_..._MS_R.TIF\n"
            "│   ├── DJI_..._MS_RE.TIF\n│   └── DJI_..._MS_NIR.TIF\n├── 2/\n│   └── ...\n└── 3/\n    └── ...",
            language=None)

subidos = st.file_uploader("ZIP con las capturas, o los 4 TIFF de una captura",
                           type=["zip", "tif", "tiff"], accept_multiple_files=True)

if not subidos:
    st.session_state.pop("capturas", None)
else:
    clave = tuple((f.name, f.size) for f in subidos)
    if st.session_state.get("clave_subida") != clave:
        tmp = tempfile.mkdtemp(prefix="harina_")
        with st.spinner("Leyendo archivos..."):
            capturas, incompletas = guardar_subida(subidos, Path(tmp))
        st.session_state.update(clave_subida=clave, capturas=capturas, incompletas=incompletas)
        st.session_state.pop("resultado", None)
        st.session_state.pop("descripcion", None)

capturas = st.session_state.get("capturas")
if capturas is not None:
    for msg in st.session_state.get("incompletas", []):
        st.warning(f"Captura incompleta, se omitirá: {msg}")
    if not capturas:
        st.error("No se encontró ninguna captura con sus 4 bandas. Revise los nombres de los archivos.")
        st.stop()
    st.success(f"Se encontraron {len(capturas)} captura(s) completa(s).")

    # --- Paso 2: describir ------------------------------------------------------
    st.header("2. Indique qué contiene cada foto")
    st.markdown(
        "Antes de procesar, indique si cada foto tiene **una muestra** o **varias en grilla**, cuántas filas y "
        "columnas, **en qué esquina está la muestra 1** y si se numera **por filas o por columnas**. "
        "Primero elija los valores generales; luego puede cambiar cualquier captura en la tabla."
    )
    d1, d2 = st.columns([3, 2])
    with d1:
        tipo = st.radio("¿Qué hay en cada foto?", TIPOS, horizontal=True)
        if tipo == TIPOS[0]:
            a, b = st.columns(2)
            filas = a.number_input("Filas de platos", 1, 12, 4)
            columnas = b.number_input("Columnas de platos", 1, 12, 4)
            a, b = st.columns(2)
            esquina = a.selectbox("La muestra 1 está en la esquina", P.ESQUINAS,
                                  help="Tal como se ve en la foto del dron, no en la mesa.")
            recorrido = b.selectbox("Se numera", P.RECORRIDOS,
                                    help="Por filas: 1, 2, 3, 4 avanzan de lado a lado. "
                                         "Por columnas: avanzan de arriba hacia abajo.")
        else:
            filas, columnas, esquina, recorrido = 1, 1, P.ESQUINAS[0], P.RECORRIDOS[0]
        inicio = st.number_input("Número de la primera muestra", 0, 10**7, 1,
                                 help="En la tabla puede poner otro número inicial por captura, o la lista "
                                      "exacta de códigos si no son correlativos.")
    with d2:
        if tipo == TIPOS[0]:
            st.markdown("**Así quedará la numeración en la foto**")
            st.dataframe(mapa_numeracion(int(filas), int(columnas), esquina, recorrido, int(inicio)),
                         use_container_width=True)
        else:
            st.info("Se tomará la zona de harina más grande de cada foto. "
                    "Si la carpeta tiene un número, ese será el número de la muestra.")

    base = tabla_descripcion(capturas, tipo, int(filas), int(columnas), esquina, recorrido, int(inicio))
    st.markdown("**Descripción por captura** (edite las celdas si alguna foto es distinta)")
    descripcion = st.data_editor(
        base, hide_index=True, use_container_width=True, disabled=["Carpeta", "Captura"],
        key=f"editor_{tipo}_{filas}_{columnas}_{esquina}_{recorrido}_{inicio}",
        column_config={
            "Tipo": st.column_config.SelectboxColumn(options=["Varias muestras (grilla)", "Una muestra"], required=True),
            "Filas": st.column_config.NumberColumn(min_value=1, max_value=12, step=1),
            "Columnas": st.column_config.NumberColumn(min_value=1, max_value=12, step=1),
            "Muestra 1 en": st.column_config.SelectboxColumn(options=P.ESQUINAS, required=True),
            "Recorrido": st.column_config.SelectboxColumn(options=P.RECORRIDOS, required=True),
            "N° inicial": st.column_config.NumberColumn(min_value=0, step=1),
            "Códigos (opcional)": st.column_config.TextColumn(
                help="Lista de códigos en el orden de numeración, separados por comas. Úsela si los lotes "
                     "no son correlativos, por ejemplo: 212, 213, 215"),
            "Excluir (opcional)": st.column_config.TextColumn(
                help="Posiciones sin plato o que no deben medirse, por ejemplo: F1C1, F4C4. "
                     "No se miden ni se numeran."),
        },
    )

    # --- Paso 3: procesar -------------------------------------------------------
    st.header("3. Procese y revise la segmentación")
    if st.button("Procesar", type="primary"):
        tablas, imagenes, avisos = [], {}, []
        barra = st.progress(0.0)
        for k, (cap, (_, fila)) in enumerate(zip(capturas, descripcion.iterrows()), start=1):
            nombre = f"{cap['Carpeta']} / {cap['Captura']}"
            barra.progress((k - 1) / len(capturas), text=f"Procesando {nombre} ({k} de {len(capturas)})")
            una = fila["Tipo"] == "Una muestra"
            cfg = {"tipo": "Una muestra" if una else "Grilla", "filas": fila["Filas"], "columnas": fila["Columnas"],
                   "esquina": fila["Muestra 1 en"], "recorrido": fila["Recorrido"], "inicio": fila["N° inicial"],
                   "excluir": P.leer_lista(fila["Excluir (opcional)"]),
                   "codigos": P.leer_lista(fila["Códigos (opcional)"]) or None}
            try:
                df, vis, av = P.procesar_captura(cap["archivos"], cap["Captura"], cfg)
                df.insert(0, "Carpeta", cap["Carpeta"])
                tablas.append(df)
                imagenes[nombre] = vis
                avisos += av
            except Exception as e:
                avisos.append(f"{nombre}: {e}")
        barra.empty()
        res = pd.concat(tablas, ignore_index=True) if tablas else pd.DataFrame()
        if "Revisar" in res:
            res.insert(res.columns.get_loc("Columna") + 1, "Revisar", res.pop("Revisar").fillna(""))
        st.session_state["resultado"] = (res, imagenes, avisos)
        st.session_state["descripcion"] = descripcion.copy()

    if "resultado" in st.session_state:
        res, imagenes, avisos = st.session_state["resultado"]
        for a in avisos:
            st.warning(a)
        if res.empty:
            st.error("No se obtuvo ninguna muestra. Revise la descripción de las capturas.")
            st.stop()

        st.markdown("Verifique en cada imagen que **el contorno verde quede solo sobre la harina** y que "
                    "**el número amarillo** corresponda a la muestra correcta. Si algo no calza, corrija la tabla "
                    "del paso 2 y vuelva a procesar.")
        nombres = list(imagenes)
        if len(nombres) == 1:
            st.image(imagenes[nombres[0]], caption=nombres[0], use_container_width=True)
        else:
            elegido = st.selectbox("Captura a revisar", nombres)
            st.image(imagenes[elegido], caption=elegido, use_container_width=True)

        # --- Paso 4: resultados ---------------------------------------------------
        st.header("4. Resultados")
        res_eval, params = evaluar(res, modelos, features, limites)
        n_rev = int((res_eval["Revisar"].fillna("") != "").sum())
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Muestras", len(res_eval))
        k2.metric("Para revisar", n_rev)
        if params:
            k3.metric("Conformes", int((res_eval["Calidad"] == "Conforme").sum()))
            k4.metric("No conformes", int((res_eval["Calidad"] == "No conforme").sum()))

        visibles = ["Carpeta", "Muestra", "Pocillo", "Revisar", "px_harina_usados",
                    "Green_rel", "Red_rel", "RedEdge_rel", "NIR_rel"]
        visibles += [c for c in res_eval if c.endswith(("estimada (%)", "cumple")) or c in ("Calidad", "Motivo")]
        # Calidad y Motivo primero, para que el porqué se vea sin desplazarse
        if params:
            visibles = ["Carpeta", "Muestra", "Pocillo", "Calidad", "Motivo"] + \
                [c for c in visibles if c not in ("Carpeta", "Muestra", "Pocillo", "Calidad", "Motivo")]

        lineas = conclusiones(res_eval, params, limites)
        if lineas:
            st.subheader("Conclusión")
            st.markdown("\n".join(f"- {l}" for l in lineas))

        def colorear(v):
            if v in ("Sí", "Conforme"):
                return "background-color: #C6EFCE; color: #006100"
            if v in ("No", "No conforme"):
                return "background-color: #FFC7CE; color: #9C0006"
            if isinstance(v, str) and v and v not in ("Sin datos",):
                return "background-color: #FFEB9C; color: #7A5200"
            return ""

        estado = [c for c in visibles if c.endswith("cumple") or c in ("Calidad", "Revisar")]
        formato = {c: "{:.4f}" for c in ("Green_rel", "Red_rel", "RedEdge_rel", "NIR_rel")}
        formato.update({c: "{:.2f}" for c in visibles if c.endswith("(%)")})
        formato["px_harina_usados"] = "{:.0f}"
        st.markdown("**Detalle por muestra**")
        st.dataframe(res_eval[visibles].style.map(colorear, subset=estado).format(formato, na_rep=""),
                     use_container_width=True, hide_index=True,
                     column_config={"Revisar": st.column_config.TextColumn(width="medium"),
                                    "Motivo": st.column_config.TextColumn(width="large")})

        info = {
            "Fecha de procesamiento": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Capturas procesadas": len(imagenes),
            "Muestras": len(res_eval),
            "Segmentación": "Otsu local por plato, erosión de 25 px del borde, recorte de percentiles 2-98",
            "Normalización": "Mediana de la harina dividida entre el blanco de la bandeja alrededor del plato (_rel)",
            "Modelo": ", ".join(modelos) if params else "Sin modelo (solo datos de bandas)",
        }
        for p, (tl, lim) in limites.items():
            info[f"Límite {p}"] = f"{'≥' if tl == 'min' else '≤'} {lim} %"
        info["Nota"] = "Valores estimados; no reemplazan el análisis de laboratorio certificado."

        b1, b2 = st.columns(2)
        b1.download_button("Descargar resultados en Excel",
                           generar_excel(res_eval, st.session_state["descripcion"], info,
                                         [re.sub(r"\*\*", "", l) for l in lineas]),
                           f"calidad_harina_{datetime.now():%Y%m%d_%H%M}.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", type="primary")
        b2.download_button("Descargar imágenes de control (ZIP)", zip_imagenes(imagenes),
                           f"segmentacion_{datetime.now():%Y%m%d_%H%M}.zip", "application/zip")

with st.expander("Qué contiene el Excel"):
    st.markdown(
        "- **Resultados:** una fila por muestra con su carpeta, número, posición en la bandeja (FxCy), "
        "una columna **Revisar** (plato vacío, harina cortada por el borde de la foto, etc.), los píxeles de harina "
        "usados y, por banda, la reflectancia relativa al blanco (`_rel`) y su variación (`_cv`). También trae las "
        "variables derivadas (logaritmos y diferencias normalizadas entre bandas) y, si hay modelo, la estimación de "
        "cada parámetro, si cumple y la calidad general.\n"
        "- **Descripción de capturas:** lo que se indicó en el paso 2 para cada foto.\n"
        "- **Información:** fecha, método y límites usados."
    )

# ---------------------------------------------------------------------------
# Información del proyecto
# ---------------------------------------------------------------------------
st.divider()
st.header("¿Cómo funciona?")
p1, p2, p3, p4 = st.columns(4)
p1.markdown("**1. Captura**\n\nSe fotografían las muestras con un dron **DJI Mavic 3 Multispectral**: "
            "Verde (560 nm), Rojo (650 nm), Red Edge (730 nm) e Infrarrojo cercano (860 nm).")
p2.markdown("**2. Calibración**\n\nSe corrige el viñeteado, el nivel negro, la ganancia y la exposición de cada "
            "banda, y se normaliza con el blanco de la bandeja que rodea cada plato.")
p3.markdown("**3. Segmentación**\n\nSe ubica cada plato y se separa solo la harina, descartando el borde, "
            "el plástico, la mesa y los píxeles extremos (brillos o polvo).")
p4.markdown("**4. Resultado**\n\nSe calculan las variables espectrales de cada muestra y, si hay modelo, "
            "se estima su calidad y se compara con la especificación.")

with st.expander("Datos y entrenamiento"):
    st.markdown(
        "- Los modelos se entrenan con **contramuestras de cerca de 300 lotes de producción**, cada una con su "
        "resultado de laboratorio certificado.\n"
        "- Se evalúan con validación **dejando una captura completa fuera**, para medir cómo funcionaría el modelo "
        "con una foto nueva.\n"
        "- La referencia metodológica es una tesis que predice estos parámetros con imágenes **hiperespectrales** "
        "e inteligencia artificial."
    )
with st.expander("Alcance y limitaciones"):
    st.markdown(
        "- El Mavic 3M registra **4 bandas**, frente a las cerca de 240 de una cámara hiperespectral, por lo que la "
        "precisión esperada es bastante menor.\n"
        "- Los resultados son **estimaciones experimentales** y no reemplazan el análisis de laboratorio certificado.\n"
        "- El proyecto está en desarrollo: los modelos se actualizan a medida que se incorporan más datos."
    )
st.caption(f"Proyectos de Mecanica Electrica · {INSTITUCION}")
