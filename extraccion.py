"""Funciones reutilizables para extraer las variables que usa el modelo web.

Este módulo recibe las bandas ya calibradas y las máscaras de harina/blanco.
La lectura TIFF y la segmentación siguen en procesamiento.py, igual que en el
flujo de entrenamiento; aquí se concentra el cálculo de las características.
"""

from itertools import combinations

import numpy as np
import pandas as pd


BANDAS = ("Green", "Red", "RedEdge", "NIR")


def extraer_caracteristicas(
    bandas: dict[str, np.ndarray],
    mascara_harina: np.ndarray,
    mascara_blanco: np.ndarray,
) -> dict[str, float]:
    """Calcula reflectancia relativa y variación para una muestra segmentada.

    El 2 % de valores más claros y más oscuros se excluye como en el
    entrenamiento, para reducir el efecto de brillos, sombras y polvo.
    """
    resumen = {}
    for banda in BANDAS:
        valores = bandas[banda][mascara_harina]
        if valores.size == 0:
            resumen[f"{banda}_rel"] = np.nan
            resumen[f"{banda}_cv"] = np.nan
            continue

        bajo, alto = np.percentile(valores, [2, 98])
        valores = valores[(valores >= bajo) & (valores <= alto)]

        blanco_local = bandas[banda][mascara_blanco]
        if blanco_local.size:
            mitad_clara = np.percentile(blanco_local, 50)
            blanco = np.median(blanco_local[blanco_local >= mitad_clara])
        else:
            blanco = np.nan

        # La división por el blanco reduce los cambios de luz entre capturas.
        resumen[f"{banda}_rel"] = float(np.median(valores) / blanco)
        # El coeficiente de variación describe cuánto cambia la señal dentro
        # de la harina en relación con su nivel medio.
        resumen[f"{banda}_cv"] = float(np.std(valores) / np.mean(valores))

    return resumen


def calcular_variables(resumen: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas *_rel y *_cv en las 14 variables del modelo web.

    El orden de salida coincide con mejorar_modelos_v2.variables() y con el
    archivo modelo_calidad.joblib entrenado desde dataset_segmentado_v3.xlsx.
    """
    relativas = {b: resumen[f"{b}_rel"].values for b in BANDAS}
    variables = pd.DataFrame(
        {f"log_{b}": np.log(relativas[b]) for b in BANDAS}, index=resumen.index
    )
    for primera, segunda in combinations(BANDAS, 2):
        a, b = relativas[primera], relativas[segunda]
        variables[f"ND_{primera}_{segunda}"] = (a - b) / (a + b)
    for banda in BANDAS:
        variables[f"cv_{banda}"] = resumen[f"{banda}_cv"].values
    return variables
