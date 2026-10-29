"""Compara la extracción de la web con la referencia del pipeline entrenado.

Usa dos capturas pequeñas del proyecto (capturas 9 y 10). Si se copia solo la
carpeta de la web y no están los datos originales, la prueba se omite.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


WEB = Path(__file__).resolve().parents[1]
PROYECTO = WEB.parent
SCRIPTS_PIPELINE = PROYECTO / "03_MODELOS_Y_ENTRENAMIENTO" / "scripts"
DATASET = PROYECTO / "02_EXTRACCION_Y_SEGMENTACION" / "datos_generados" / "dataset_segmentado_v3.xlsx"
CAPTURAS = PROYECTO / "01_DATOS_ORIGINALES" / "capturas_dron"


def _cargar_modulos():
    # El módulo de entrenamiento carga primero sus propias funciones de grilla.
    sys.path.insert(0, str(SCRIPTS_PIPELINE))
    import mejorar_modelos_v2 as entrenamiento

    sys.path.insert(0, str(WEB))
    import procesamiento as web

    return entrenamiento, web


def _iou(caja_a, caja_b):
    x0, y0, x1, y1 = caja_a
    a0, b0, a1, b1 = caja_b
    ancho = max(0, min(x1, a1) - max(x0, a0))
    alto = max(0, min(y1, b1) - max(y0, b0))
    interseccion = ancho * alto
    union = (x1 - x0) * (y1 - y0) + (a1 - a0) * (b1 - b0) - interseccion
    return interseccion / union if union else 0.0


class ComparacionConEntrenamiento(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        requeridos = [DATASET]
        requeridos += [CAPTURAS / f"captura_{n}" for n in (9, 10)]
        if not all(ruta.exists() for ruta in requeridos):
            raise unittest.SkipTest("No están disponibles las capturas y el dataset del pipeline.")

        cls.entrenamiento, cls.web = _cargar_modulos()
        cls.dataset = pd.read_excel(DATASET)
        cls.caracteristicas = [
            columna for columna in cls.dataset.columns
            if columna.startswith(("log_", "ND_", "cv_"))
        ]
        cls.resultados = {}

        for numero in (9, 10):
            carpeta = CAPTURAS / f"captura_{numero}"
            prefijo, archivos = next(iter(cls.web.buscar_archivos(carpeta).items()))
            referencia = cls.dataset[
                (cls.dataset["Captura"] == prefijo) & cls.dataset["seg_ok"].astype(bool)
            ].set_index("Pocillo")

            # Reconstruye las cajas del entrenamiento desde la grilla guardada,
            # usando el emparejamiento con las muestras del dataset original.
            _, cajas_entrenamiento = cls.entrenamiento.emparejar_recuadros(
                carpeta, prefijo, referencia.reset_index()
            )

            # La web detecta la grilla automáticamente sobre la banda Green.
            _, green = cls.web.senal_sin_irradiancia(archivos["Green"])
            cajas_web = cls.web.detectar_grilla(green, 4, 4)

            # Las etiquetas pueden empezar desde esquinas distintas; se empareja
            # cada muestra por su posición real y el solapamiento de las cajas.
            cajas_detectadas = list(cajas_web)
            etiquetas_pipeline = list(cajas_entrenamiento)
            solapamientos = np.array([
                [_iou(cajas_web[c], cajas_entrenamiento[e]) for e in etiquetas_pipeline]
                for c in cajas_detectadas
            ])
            filas, columnas = linear_sum_assignment(1.0 - solapamientos)
            correspondencia = {
                cajas_detectadas[i]: etiquetas_pipeline[j]
                for i, j in zip(filas, columnas)
            }
            ious = [solapamientos[i, j] for i, j in zip(filas, columnas)]

            cfg = {
                "tipo": "Grilla", "filas": 4, "columnas": 4,
                "esquina": cls.web.ESQUINAS[0], "recorrido": cls.web.RECORRIDOS[0],
                "inicio": 1, "excluir": [],
            }
            resultado_web, _, avisos = cls.web.procesar_captura(archivos, prefijo, cfg)
            resultado_web = resultado_web.set_index("Pocillo")

            diferencias = {}
            for caracteristica in cls.caracteristicas:
                porcentajes = []
                for (fila, columna), etiqueta_pipeline in correspondencia.items():
                    etiqueta_web = f"F{fila + 1}C{columna + 1}"
                    valor_pipeline = float(referencia.loc[etiqueta_pipeline, caracteristica])
                    valor_web = float(resultado_web.loc[etiqueta_web, caracteristica])
                    # El divisor protege el cálculo si alguna característica de
                    # referencia es cero; la mediana reduce el efecto de valores
                    # muy pequeños en algunos índices normalizados.
                    porcentajes.append(
                        abs(valor_web - valor_pipeline)
                        / max(abs(valor_pipeline), 1e-12) * 100
                    )
                diferencias[caracteristica] = np.asarray(porcentajes)

            cls.resultados[numero] = {
                "iou": np.asarray(ious),
                "segmentadas": int(resultado_web["seg_ok"].fillna(False).sum()),
                "avisos": avisos,
                "diferencias": diferencias,
            }

    def test_deteccion_y_variables_se_mantienen_cerca_del_pipeline(self):
        for numero, datos in self.resultados.items():
            with self.subTest(captura=numero):
                self.assertEqual(datos["segmentadas"], 16)
                self.assertFalse(datos["avisos"])
                self.assertGreaterEqual(float(np.median(datos["iou"])), 0.90)
                for nombre, porcentajes in datos["diferencias"].items():
                    # La comparación manual observó medianas menores a 0.22 %.
                    # Estos límites dejan margen a diferencias menores entre
                    # versiones de OpenCV sin aceptar un desplazamiento grande.
                    self.assertLess(float(np.median(porcentajes)), 1.5, nombre)
                    self.assertLess(float(np.max(porcentajes)), 12.0, nombre)


if __name__ == "__main__":
    unittest.main()
