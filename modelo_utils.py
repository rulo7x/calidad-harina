"""Pequeñas utilidades para identificar el tipo del paquete de modelo."""


def es_modelo_demostracion(paquete: dict) -> bool:
    """Dice si el paquete está marcado como modelo de prueba.

    Los paquetes reales existentes no tienen una marca y se consideran reales.
    Para mostrar el aviso, un paquete de demostración debe incluir
    ``"tipo": "prueba"`` o ``"modelo": "prueba"``.
    """
    tipo = paquete.get("tipo", paquete.get("modelo", "real"))
    return str(tipo).strip().casefold() in {"prueba", "demo", "demostracion", "demostración"}
