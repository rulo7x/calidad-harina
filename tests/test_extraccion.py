"""Pruebas sencillas de la extracción usando TIFF de 16 bits sintéticos."""

from io import BytesIO
import unittest

import numpy as np
import pandas as pd
from PIL import Image

from extraccion import calcular_variables, extraer_caracteristicas


class PruebasExtraccion(unittest.TestCase):
    def test_tiff_sinteticos_producen_valores_conocidos(self):
        # Cada imagen tiene un centro uniforme (harina) y un borde uniforme
        # (blanco de referencia), así el resultado esperado se calcula a mano.
        valores_harina = {"Green": 2000, "Red": 3000, "RedEdge": 4000, "NIR": 5000}
        valores_blanco = {"Green": 10000, "Red": 10000, "RedEdge": 10000, "NIR": 10000}
        mascara_harina = np.zeros((5, 5), dtype=bool)
        mascara_harina[1:4, 1:4] = True
        mascara_blanco = ~mascara_harina

        rutas = {}
        for banda, valor in valores_harina.items():
            imagen = np.full((5, 5), valores_blanco[banda], dtype=np.uint16)
            imagen[mascara_harina] = valor
            # El TIFF se crea en memoria: probamos el formato sin dejar
            # archivos abiertos o temporales en Windows.
            with BytesIO() as contenido:
                Image.fromarray(imagen).save(contenido, format="TIFF")
                contenido.seek(0)
                with Image.open(contenido) as archivo:
                    rutas[banda] = np.array(archivo, dtype=np.float64, copy=True)

        resumen = extraer_caracteristicas(rutas, mascara_harina, mascara_blanco)
        esperados = {"Green": 0.2, "Red": 0.3, "RedEdge": 0.4, "NIR": 0.5}
        for banda, relativo in esperados.items():
            self.assertAlmostEqual(resumen[f"{banda}_rel"], relativo, places=7)
            self.assertAlmostEqual(resumen[f"{banda}_cv"], 0.0, places=7)

        variables = calcular_variables(pd.DataFrame([resumen]))
        self.assertEqual(
            list(variables.columns),
            [
                "log_Green", "log_Red", "log_RedEdge", "log_NIR",
                "ND_Green_Red", "ND_Green_RedEdge", "ND_Green_NIR",
                "ND_Red_RedEdge", "ND_Red_NIR", "ND_RedEdge_NIR",
                "cv_Green", "cv_Red", "cv_RedEdge", "cv_NIR",
            ],
        )
        self.assertAlmostEqual(variables.loc[0, "log_Green"], np.log(0.2))
        self.assertAlmostEqual(variables.loc[0, "ND_Green_Red"], -0.2)


if __name__ == "__main__":
    unittest.main()
