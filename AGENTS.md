# AGENTS.md — Web de análisis de harina de pescado

## Propósito

Esta carpeta contiene la aplicación web Streamlit del proyecto PME Grupo 09. Recibe imágenes multiespectrales del dron DJI Mavic 3M y procesa muestras de harina para estimar parámetros de calidad y asignar un grado comercial. La interfaz y los textos están en español.

El flujo real admite una captura de una muestra o una bandeja con varias muestras. La aplicación actual trabaja con las bandas Green, Red, Red Edge y NIR, y puede procesar varias capturas organizadas en carpetas dentro de un ZIP.

## Tecnología y ejecución

- Python.
- Streamlit para la interfaz y el flujo de carga, revisión y descarga.
- NumPy, pandas, OpenCV y Pillow para procesar imágenes y datos; scikit-learn y joblib para los modelos; openpyxl para los reportes Excel.
- La lista de dependencias está en `requirements.txt`.

Para iniciar la web:

```powershell
pip install -r requirements.txt
streamlit run app_calidad.py
```

## Organización

- `app_calidad.py`: página principal, carga de archivos, configuración de capturas, predicción, clasificación y descargas.
- `procesamiento.py`: lectura y calibración de bandas, detección de muestras o grillas, segmentación de harina y cálculo de variables.
- `extraccion.py`: extracción reutilizable de reflectancias relativas, variación y variables derivadas que recibe el modelo.
- `modelo_utils.py`: identifica paquetes marcados como modelo de demostración para mostrar el aviso correspondiente.
- `analisis_harina_pescado.py`: funciones de calibración radiométrica y detección de grilla reutilizadas por el procesamiento web.
- `diseno.py`: estilos, presentación y secciones informativas de la página.
- `modelo_calidad.joblib`: paquete de modelos y nombres de características que carga la aplicación cuando está disponible.
- `assets/`: imágenes de la página.
- `tests/`: pruebas de la extracción con TIFF sintéticos, del aviso de demostración y comparación de la detección web con capturas del entrenamiento cuando los datos fuente están disponibles.
- `.streamlit/config.toml`: configuración de Streamlit, incluida la apariencia y el límite de subida.
- `version_anterior/`: versión anterior de la web, conservada como referencia; no es el punto de entrada actual.

## Flujo de procesamiento actual

1. El usuario sube cuatro TIFF por captura (`_MS_G`, `_MS_R`, `_MS_RE`, `_MS_NIR`), sueltos o dentro de un ZIP con carpetas por captura.
2. La página agrupa los archivos por sufijo y permite indicar si cada captura contiene una muestra o una grilla, además de su numeración y exclusiones.
3. `procesamiento.py` lee los TIFF y aplica la calibración radiométrica del proyecto. Para cada muestra localiza la zona de harina y segmenta los píxeles de harina, separándolos del plato y el fondo.
4. Calcula por banda medidas de reflectancia relativa al blanco local y de variación, además de variables derivadas entre bandas. Estas características se usan con el modelo guardado en `modelo_calidad.joblib` cuando el archivo y sus variables están disponibles.
5. La página muestra imágenes de control y resultados por muestra, clasifica según las especificaciones comerciales configuradas y permite descargar un Excel y un ZIP de imágenes de control.

La extracción usada en la página debe mantenerse alineada con el procesamiento y las características del modelo real del proyecto. No sustituirla por un promedio de una ROI central: ese flujo pertenece a una versión anterior y no representa el procedimiento actual.

## Modelo y resultados

El modelo actual se carga desde `modelo_calidad.joblib`. La aplicación espera un diccionario que incluya `modelos` y `features`; cada modelo predice sus parámetros a partir de las columnas indicadas por `features`. El campo opcional `tipo` (o `modelo`) puede valer `"prueba"` para marcar un paquete de demostración y mostrar el aviso correspondiente. El paquete real actual no tiene esa marca y se interpreta como real. La página puede funcionar sin el archivo y presentar datos de bandas sin predicciones.

La clasificación actual usa los parámetros predichos disponibles (proteína, grasa, humedad y ceniza) y los grados comerciales definidos en `app_calidad.py`. Debe indicarse claramente que las predicciones son experimentales y no reemplazan el análisis de laboratorio certificado. No presentar un modelo como entrenado con datos reales si el archivo cargado es de demostración.

## Reglas para cambios

- Mantener Streamlit, la segmentación por muestra y la clasificación comercial del flujo actual, salvo que el usuario pida otra cosa.
- Mantener simple el código y explicar en español por qué se toman decisiones importantes.
- No modificar archivos fuera de esta carpeta.
- Limitar y validar las subidas. Las imágenes de usuario deben permanecer solo el tiempo necesario para procesarlas y deben eliminarse al terminar.
- Al cambiar extracción o predicción, comprobar que las características coincidan con las usadas para entrenar el modelo. Si no es posible establecerlo, explicar las diferencias antes de asumir equivalencia.
- Agregar pruebas para el procesamiento que se modifique, incluyendo imágenes sintéticas cuando sea práctico.
- Si el usuario pide completar un cambio, ejecutar las pruebas pertinentes y comprobar que la aplicación inicia localmente.
