from noshow.registry import EXPERIMENTOS


def test_registro_tem_campos_obrigatorios():
    for v, cfg in EXPERIMENTOS.items():
        assert {"titulo", "mudanca", "base", "categoricas", "continuas"} <= set(cfg), v
