import importlib.util
from pathlib import Path

ESPEC = importlib.util.spec_from_file_location(
    "rastreabilidade", Path(__file__).resolve().parents[1] / "scripts" / "rastreabilidade.py")
rastreabilidade = importlib.util.module_from_spec(ESPEC)
ESPEC.loader.exec_module(rastreabilidade)


def test_rastreabilidade_valida_a_fonte_unica():
    pendencias = []
    assert rastreabilidade.validar(rastreabilidade.carregar(), pendencias) == []
    assert all("implementação" in p for p in pendencias)  # só artefatos de issues em andamento podem faltar


def test_validacao_acusa_repeticao_teste_inexistente_e_cenario_alheio():
    issues = {1: {"pr": 1, "cenarios": {1: ["RF-009"]}, "requisitos": [
        {"id": "RF-001", "texto": "a", "implementacao": ["src/noshow/data.py::load_raw"], "verificacao": ["tests::nao_existe"]},
        {"id": "RF-001", "texto": "b", "implementacao": ["src/noshow/data.py::funcao_inexistente"], "verificacao": ["x"]},
    ]}}
    erros = " | ".join(rastreabilidade.validar(issues))
    assert "repetidos" in erros and "teste não encontrado" in erros
    assert "implementação não encontrada" in erros and "não pertence à issue" in erros


def test_reescrita_preserva_marcacao_e_e_idempotente():
    issue = {"pr": 15, "cenarios": {1: ["RF-001"]}, "requisitos": [
        {"id": "RF-001", "texto": "Carregar", "implementacao": ["src/noshow/data.py::load_raw"], "verificacao": ["notebook 01"]},
        {"id": "RF-002", "texto": "Converter", "implementacao": ["src/noshow/data.py::load_raw"], "verificacao": ["notebook 01"]},
    ]}
    corpo = ("### Requisitos\n\n- [x] RF01: Carregar\n- [ ] RF02: Converter\n\n### Critérios de Aceite\n\n"
             "**Cenário 1:** Carga\n- **Dado** algo")
    novo = rastreabilidade.reescrever_corpo(corpo, issue)
    assert "- [x] **RF-001** Carregar" in novo and "- [ ] **RF-002** Converter" in novo
    assert "↳ Implementação: `src/noshow/data.py::load_raw`" in novo and "Entrega: #15" in novo
    assert "**Cenário 1:** Carga · verifica RF-001" in novo
    assert rastreabilidade.reescrever_corpo(novo, issue) == novo
