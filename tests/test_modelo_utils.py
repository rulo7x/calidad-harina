import unittest

from modelo_utils import es_modelo_demostracion


class PruebasTipoModelo(unittest.TestCase):
    def test_paquete_marcado_como_prueba_muestra_aviso(self):
        self.assertTrue(es_modelo_demostracion({"tipo": "prueba"}))
        self.assertTrue(es_modelo_demostracion({"modelo": "DEMO"}))

    def test_paquete_real_o_sin_marca_no_muestra_aviso(self):
        self.assertFalse(es_modelo_demostracion({"tipo": "real"}))
        self.assertFalse(es_modelo_demostracion({"modelos": {}, "features": []}))


if __name__ == "__main__":
    unittest.main()
