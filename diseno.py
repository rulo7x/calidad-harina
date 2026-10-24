"""
diseno.py - Estilos y secciones de presentación de la página (portada, pasos, cifras,
hallazgos y pie). La lógica de análisis está en app_calidad.py y procesamiento.py.
"""

import base64
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "assets"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

:root{ --verde:#2FB45A; --verde-h:#26994B; --verde-osc:#0D4A32; --menta:#B8EBCB;
       --texto:#2D2D2D; --gris:#5E6B66; --suave:#F4F6F5; --borde:#E7ECE9; }

.stApp *:not(code):not(pre):not([data-testid="stIconMaterial"]):not([class*="material"]) { font-family:'Poppins', sans-serif !important; }
[data-testid="stHeaderActionElements"]{ display:none !important; }
.stMainBlockContainer > [data-testid="stVerticalBlock"]{ gap:0 !important; }
.v{ color:var(--verde) !important; }
.stApp, [data-testid="stAppViewContainer"]{ background:#FFFFFF; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"], [data-testid="stStatusWidget"]{ display:none !important; }
header[data-testid="stHeader"]{ display:none; }
.block-container{ max-width:100% !important; padding:0 !important; }
.stMarkdown p, .stMarkdown li{ color:var(--texto); }
a{ text-decoration:none !important; }
.cont{ max-width:1180px; margin:0 auto; padding:0 28px; }

/* barra superior y navegación */
.aviso-top{ background:linear-gradient(90deg,#0D4A32,#1E7F45); color:#fff; text-align:center; font-size:14px;
  padding:9px 12px; font-weight:400; }
.nav{ background:#fff; border-bottom:1px solid var(--borde); }
.nav .cont{ display:flex; align-items:center; justify-content:space-between; height:74px; }
.logo{ font-weight:700; font-size:22px; letter-spacing:-.5px; color:var(--texto); }

.menu{ display:flex; align-items:center; gap:30px; }
.menu a{ color:var(--texto); font-size:15px; font-weight:400; }
.menu a:hover{ color:var(--verde); }
.pill{ display:inline-block; background:var(--verde); color:#fff !important; font-weight:500; font-size:15px;
  padding:10px 28px; border-radius:999px; transition:background .2s; }
.pill:hover{ background:var(--verde-h); }
.pill.grande{ font-size:16px; padding:14px 40px; }

/* portada */
.hero{ display:grid; grid-template-columns:1fr 1.05fr; gap:56px; align-items:center; padding:70px 0 80px; }
.hero h1{ font-size:54px; line-height:1.15; font-weight:500; letter-spacing:-1.6px; color:var(--texto); margin:0 0 30px; }

.hero p{ font-size:17px; color:var(--gris); line-height:1.7; margin:0 0 34px; max-width:500px; }
.hero-foto{ position:relative; }
.hero-foto img{ width:100%; border-radius:28px; display:block; }
.chip{ position:absolute; background:#fff; border-radius:999px; padding:8px 16px 8px 10px; font-size:13px; font-weight:500;
  color:var(--texto); box-shadow:0 8px 24px rgba(0,0,0,.12); display:flex; align-items:center; gap:8px; }
.chip i{ width:22px; height:22px; border-radius:50%; background:var(--verde); display:inline-block; position:relative; }
.chip i:after{ content:""; position:absolute; left:8px; top:4px; width:5px; height:10px; border:solid #fff; border-width:0 2px 2px 0; transform:rotate(45deg); }
.chip.c1{ top:-16px; left:28px; } .chip.c2{ bottom:26px; right:-14px; }

/* sección con foto a todo lo ancho */
.banda-foto{ background-size:cover; background-position:center; position:relative; padding:90px 0 70px; }
.banda-foto:before{ content:""; position:absolute; inset:0; background:linear-gradient(180deg,rgba(13,74,50,.55),rgba(13,74,50,.88)); }
.banda-foto .cont{ position:relative; }
.vidrio{ max-width:560px; background:rgba(255,255,255,.16); backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px);
  border:1px solid rgba(255,255,255,.25); border-radius:26px; padding:34px 36px; color:#fff; }
.vidrio h2{ color:#fff; font-size:34px; font-weight:500; line-height:1.25; margin:0 0 14px; letter-spacing:-.8px; }
.vidrio p{ color:#E9F5EE; font-size:16px; line-height:1.7; margin:0; }
.pasos{ display:grid; grid-template-columns:repeat(4,1fr); margin-top:64px; text-align:center; }
.pasos .p{ color:#fff; }
.pasos .p svg{ display:block; margin:0 auto 12px; }
.pasos .p b{ display:block; font-weight:500; font-size:17px; margin-bottom:6px; color:#fff; }
.pasos .p small{ display:block; color:#D7EBDF; font-size:13.5px; line-height:1.55; padding:0 14px; }
.linea{ display:flex; align-items:center; margin:34px auto 0; max-width:780px; }
.linea .pt{ width:14px; height:14px; border-radius:50%; background:#fff; flex:none; }
.linea .pt.on{ width:34px; height:34px; border:2px solid #fff; background:transparent; display:flex; align-items:center; justify-content:center; }
.linea .tr{ height:2px; background:rgba(255,255,255,.7); flex:1; }

/* cifras */
.cifras{ background:var(--menta); }
.cifras .cont{ display:grid; grid-template-columns:1fr 1.4fr; gap:40px; align-items:center; padding:70px 28px; }
.cifras h2{ font-size:40px; font-weight:500; line-height:1.2; letter-spacing:-1px; color:var(--verde-osc); margin:0; }
.nums{ display:grid; grid-template-columns:1fr 1fr; gap:40px 30px; }
.num{ display:flex; gap:16px; align-items:flex-start; }
.num .n{ font-size:58px; font-weight:600; line-height:1; color:var(--verde-osc); letter-spacing:-2px; }
.num .t{ font-size:17px; color:var(--verde-osc); line-height:1.3; margin-top:8px; font-weight:500; }

/* títulos de sección */
.titulo{ padding:90px 0 34px; }
.titulo h2{ font-size:44px; font-weight:500; letter-spacing:-1.3px; line-height:1.2; color:var(--texto); margin:0 0 14px; }
.titulo p{ font-size:17px; color:var(--gris); max-width:680px; line-height:1.7; margin:0; }

/* bloque oscuro de hallazgos */
.oscuro{ background:radial-gradient(1200px 500px at 85% 10%, #1D7A48 0%, #0D4A32 55%, #0A3A27 100%); margin-top:90px; }
.oscuro .cont{ display:grid; grid-template-columns:1fr 1.5fr; gap:60px; padding:90px 28px; }
.oscuro h2{ color:#fff; font-size:42px; font-weight:500; line-height:1.2; letter-spacing:-1.2px; margin:0 0 20px; }
.oscuro .lado p{ color:#CFE6D8; font-size:16px; line-height:1.7; }
.grid-h{ display:grid; grid-template-columns:1fr 1fr; gap:42px 40px; }
.h-item{ display:flex; gap:16px; }
.h-item svg{ flex:none; }
.h-item b{ display:block; color:#9BE58A; font-size:19px; font-weight:600; line-height:1.25; margin-bottom:8px; }
.h-item p{ color:#E3F1E8; font-size:14px; line-height:1.65; margin:0; }

/* llamado final y pie */
.cta{ background:var(--suave); border-radius:30px; display:grid; grid-template-columns:1fr 1fr; overflow:hidden; margin:90px 0; }
.cta img{ width:100%; height:100%; object-fit:cover; display:block; min-height:280px; }
.cta div{ padding:50px 48px; display:flex; flex-direction:column; justify-content:center; align-items:flex-start; }
.cta h3{ font-size:34px; font-weight:500; line-height:1.25; letter-spacing:-.8px; margin:0 0 26px; color:var(--texto); }

.pie{ background:var(--suave); }
.pie .cont{ display:grid; grid-template-columns:1.4fr 1fr 1fr 1fr; gap:30px; padding:56px 28px 30px; }
.pie h5{ font-size:14px; font-weight:600; color:var(--texto); margin:0 0 12px; }
.pie .col a, .pie .col span{ display:block; color:var(--gris); font-size:14px; margin-bottom:8px; }
.pie .legal{ grid-column:1/-1; border-top:1px solid #DCE3DF; padding-top:22px; color:#8A9690; font-size:13px; }

/* herramienta (componentes de Streamlit) */
.st-key-herramienta{ max-width:1124px; margin:0 auto 10px !important; background:#fff;
  border:1px solid var(--borde) !important; border-radius:26px !important; box-shadow:0 18px 50px -24px rgba(13,74,50,.25);
  padding:18px 26px 26px !important; }
.stButton > button, .stDownloadButton > button{ border-radius:999px !important; font-weight:500 !important; padding:.6rem 1.6rem !important; }
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"]{ background:var(--verde) !important; border-color:var(--verde) !important; color:#fff !important; }
[data-testid="stFileUploaderDropzone"]{ background:#F7FAF8; border:1.5px dashed #BFD3C7; border-radius:18px; }
h3.paso-t{ font-size:22px !important; font-weight:600 !important; color:var(--texto) !important; margin-top:24px !important; }

@media (max-width: 900px){
  .menu a:not(.pill){ display:none; }
  .hero{ grid-template-columns:1fr; padding:40px 0 50px; gap:36px; }
  .hero h1{ font-size:36px; letter-spacing:-1px; }
  .chip.c2{ right:10px; }
  .vidrio h2{ font-size:26px; }
  .pasos{ grid-template-columns:1fr 1fr; row-gap:30px; }
  .linea{ display:none; }
  .cifras .cont, .oscuro .cont, .cta, .pie .cont{ grid-template-columns:1fr; }
  .nums{ gap:28px 18px; } .num .n{ font-size:42px; }
  .titulo h2, .oscuro h2, .cifras h2{ font-size:32px; }
  .grid-h{ grid-template-columns:1fr; }
  .cta div{ padding:34px 28px; }
  .st-key-herramienta{ margin:0 12px 10px !important; padding:12px 14px 20px !important; }
}
</style>
"""


def _svg(cuerpo, color="#fff", t=46):
    return (f'<svg width="{t}" height="{t}" viewBox="0 0 48 48" fill="none" stroke="{color}" stroke-width="1.8" '
            f'stroke-linecap="round" stroke-linejoin="round">{cuerpo}</svg>')


IC = {
    "dron": '<circle cx="10" cy="10" r="5"/><circle cx="38" cy="10" r="5"/><circle cx="10" cy="38" r="5"/><circle cx="38" cy="38" r="5"/><path d="M14 14l5 5M34 14l-5 5M14 34l5-5M34 34l-5-5"/><rect x="18" y="18" width="12" height="12" rx="3"/>',
    "luz": '<circle cx="24" cy="22" r="8"/><path d="M24 4v4M24 36v4M6 22h4M38 22h4M11 9l3 3M37 9l-3 3"/><path d="M18 44h12"/>',
    "seg": '<rect x="6" y="6" width="15" height="15" rx="3"/><rect x="27" y="6" width="15" height="15" rx="3"/><rect x="6" y="27" width="15" height="15" rx="3"/><rect x="27" y="27" width="15" height="15" rx="3" stroke-dasharray="3 3"/>',
    "excel": '<path d="M28 5H12a3 3 0 0 0-3 3v32a3 3 0 0 0 3 3h24a3 3 0 0 0 3-3V16z"/><path d="M28 5v11h11M16 26h16M16 33h16M24 22v15"/>',
    "grafico": '<path d="M6 42h36"/><path d="M12 36V24M20 36V16M28 36V22M36 36V10"/>',
    "check": '<circle cx="24" cy="24" r="18"/><path d="M16 24l6 6 11-12"/>',
    "lupa": '<circle cx="21" cy="21" r="13"/><path d="M31 31l11 11M15 21h12"/>',
    "ondas": '<path d="M4 24c4-8 8-8 12 0s8 8 12 0 8-8 12 0"/><path d="M4 34c4-6 8-6 12 0s8 6 12 0 8-6 12 0"/>',
    "flecha": '<path d="M10 24h28M28 14l10 10-10 10"/>',
    "tubo": '<path d="M18 4h12M20 4v14L9 38a4 4 0 0 0 3.5 6h23a4 4 0 0 0 3.5-6L28 18V4"/><path d="M14 30h20"/>',
}


def _b64(nombre):
    ruta = ASSETS / nombre
    return ("data:image/jpeg;base64," + base64.b64encode(ruta.read_bytes()).decode()) if ruta.exists() else ""


def estilos():
    st.markdown(CSS, unsafe_allow_html=True)


def portada(institucion):
    img = _b64("portada.jpg")
    st.markdown(f"""
<div class="aviso-top">Proyecto PME · Grupo 09 · {institucion}</div>
<div class="nav"><div class="cont">
  <div class="logo">harina<span class="v">scan</span></div>
  <div class="menu">
    <a href="#como-funciona">Cómo funciona</a>
    <a href="#resultados">Resultados</a>
    <a href="#analizar">Herramienta</a>
    <a class="pill" href="#analizar">Analizar muestras</a>
  </div>
</div></div>
<div class="cont"><div class="hero">
  <div>
    <h1>Controle la calidad de su harina de pescado <span class="v">con una sola foto de dron</span></h1>
    <p>Fotografiamos las muestras con un dron multiespectral, separamos la harina de todo lo demás y estimamos
    proteína, grasa, humedad y cenizas antes de que llegue el resultado del laboratorio.</p>
    <a class="pill grande" href="#analizar">Empieza ahora</a>
  </div>
  <div class="hero-foto">
    <img src="{img}" alt="Bandeja con muestras de harina segmentadas"/>
    <div class="chip c1"><i></i>16 muestras detectadas</div>
    <div class="chip c2"><i></i>Solo la harina, sin bordes</div>
  </div>
</div></div>
""", unsafe_allow_html=True)


def como_funciona():
    fondo = _b64("fondo.jpg")
    pasos = [("dron", "Captura", "Dron DJI Mavic 3M en verde, rojo, red edge e infrarrojo"),
             ("luz", "Calibración", "Corregimos la luz con el blanco de la propia bandeja"),
             ("seg", "Segmentación", "Cada plato por separado, solo la harina"),
             ("excel", "Resultado", "Un Excel con cada muestra, su estimación y el motivo")]
    items = "".join(f'<div class="p">{_svg(IC[i])}<b>{t}</b><small>{d}</small></div>' for i, t, d in pasos)
    st.markdown(f"""
<div class="banda-foto" id="como-funciona" style="background-image:url('{fondo}')"><div class="cont">
  <div class="vidrio">
    <h2>De la bandeja al Excel en cuatro pasos</h2>
    <p>Usamos exactamente el mismo proceso con el que entrenamos el modelo, para que los números de sus fotos
    sean comparables con los de laboratorio.</p>
  </div>
  <div class="pasos">{items}</div>
  <div class="linea"><div class="pt on">{_svg(IC['flecha'], t=18)}</div><div class="tr"></div><div class="pt"></div>
    <div class="tr"></div><div class="pt"></div><div class="tr"></div><div class="pt"></div></div>
</div></div>
<div class="cifras"><div class="cont">
  <h2>Un proyecto construido con datos reales de planta</h2>
  <div class="nums">
    <div class="num">{_svg(IC['tubo'], '#0D4A32', 40)}<div><div class="n">285</div><div class="t">muestras con<br>resultado SGS</div></div></div>
    <div class="num">{_svg(IC['dron'], '#0D4A32', 40)}<div><div class="n">19</div><div class="t">capturas<br>con dron</div></div></div>
    <div class="num">{_svg(IC['ondas'], '#0D4A32', 40)}<div><div class="n">4</div><div class="t">bandas<br>espectrales</div></div></div>
    <div class="num">{_svg(IC['grafico'], '#0D4A32', 40)}<div><div class="n">11</div><div class="t">modelos<br>comparados</div></div></div>
  </div>
</div></div>
""", unsafe_allow_html=True)


def cabecera_herramienta():
    st.markdown("""
<div class="cont" id="analizar"><div class="titulo">
  <h2>Analice sus muestras <span class="v">en minutos</span></h2>
  <p>Suba las capturas del dron, indique qué hay en cada foto y revise la segmentación antes de descargar
  el Excel. Todo el procesamiento ocurre en esta página.</p>
</div></div>
""", unsafe_allow_html=True)


def hallazgos():
    v = "#9BE58A"
    items = [
        ("grafico", "R² de 0.136 en cenizas", "El mejor modelo fue PLSR de 4 componentes. Cenizas es el parámetro con más señal: error típico de 1.01 puntos frente a una variación natural de 1.09."),
        ("luz", "La luz pesaba más que la harina", "En la banda verde, la foto explicaba el 94 % de la variación. Comparar cada plato con el blanco de su bandeja lo bajó a 22 %."),
        ("seg", "Solo píxeles de harina", "Hasta un 35 % de cada recuadro era plástico, borde o mesa. Ahora se mide únicamente la harina."),
        ("check", "Validación exigente", "Cada foto se predice con un modelo que nunca la vio. Así medimos cómo funcionaría con una captura nueva."),
        ("ondas", "El límite está en el sensor", "Proteína, grasa y agua se detectan mejor entre 1100 y 2500 nm. El Mavic 3M llega hasta 860 nm."),
        ("lupa", "Siguiente paso", "Un sensor NIR o hiperespectral, captura estandarizada y más rumas con resultado de laboratorio."),
    ]
    grid = "".join(f'<div class="h-item">{_svg(IC[i], v, 40)}<div><b>{t}</b><p>{d}</p></div></div>' for i, t, d in items)
    st.markdown(f"""
<div class="oscuro" id="resultados"><div class="cont">
  <div class="lado">
    <h2>Lo que aprendimos con 285 muestras</h2>
    <p>Preferimos mostrar los números como son. Las estimaciones son experimentales y no reemplazan el
    análisis de laboratorio certificado; el veredicto de calidad se marca como parcial hasta confirmar los
    límites de la planta.</p>
  </div>
  <div class="grid-h">{grid}</div>
</div></div>
""", unsafe_allow_html=True)


def cierre(institucion):
    img = _b64("portada.jpg")
    st.markdown(f"""
<div class="cont"><div class="cta">
  <img src="{img}" alt="Muestras segmentadas"/>
  <div><h3>¿Tiene capturas nuevas? <span class="v">Súmelas al proyecto.</span></h3>
  <a class="pill grande" href="#analizar">Empieza ahora</a></div>
</div></div>
<div class="pie"><div class="cont">
  <div class="col"><div class="logo">harina<b class="v">scan</b></div>
    <span style="margin-top:12px">Control de calidad de harina de pescado con imágenes multiespectrales.</span></div>
  <div class="col"><h5>Proyecto</h5><a href="#como-funciona">Cómo funciona</a><a href="#resultados">Resultados</a><a href="#analizar">Herramienta</a></div>
  <div class="col"><h5>Datos</h5><span>DJI Mavic 3 Multispectral</span><span>Resultados SGS</span><span>285 muestras</span></div>
  <div class="col"><h5>Equipo</h5><span>PME · Grupo 09</span><span>{institucion}</span></div>
  <div class="legal">© 2026 PME Grupo 09 · {institucion}. Estimaciones experimentales con fines de investigación.</div>
</div></div>
""", unsafe_allow_html=True)
