"""Rastreabilidade dos requisitos a partir da fonte única `requisitos.yml`.

Uso:
    .venv/bin/python scripts/rastreabilidade.py                 # valida e gera REQUISITOS.md
    .venv/bin/python scripts/rastreabilidade.py --sincronizar   # também reescreve requisitos e cenários nas issues

A validação falha se houver número repetido ou fora de sequência, teste, função ou arquivo inexistente, ou
cenário citando requisito de outra issue.
"""

import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

RAIZ = Path(__file__).resolve().parents[1]
REPO = "engPisa/noshow-bayesian-framework"
PREFIXOS = {"RF": "funcional", "RNF": "não funcional", "CA": "critério de tarefa técnica"}
SECOES = ("### Requisitos", "### Critérios de aceitação")


def carregar(caminho: Path = RAIZ / "requisitos.yml") -> dict:
    return yaml.safe_load(caminho.read_text())["issues"]


def _testes_existentes() -> dict[str, str]:
    testes = {}
    for f in sorted((RAIZ / "tests").glob("test_*.py")):
        for nome in re.findall(r"^def (test_\w+)", f.read_text(), re.M):
            testes[nome] = f"tests/{f.name}"
    return testes


def _alvo_existe(alvo: str) -> bool:
    """`caminho::nome` exige o nome definido no arquivo; `caminho (...)` ou `caminho §x` exige só o arquivo."""
    caminho, _, nome = alvo.partition("::")
    caminho = re.split(r" [(§A]", caminho)[0].strip()
    arquivo = RAIZ / caminho
    if "/" not in caminho and not caminho.endswith((".md", ".yml")):
        return True  # descrição livre (ex.: "kernel do projeto")
    if "vN" in caminho:
        return any(arquivo.parent.parent.glob(f"v*/{arquivo.name}"))
    if not arquivo.exists():
        return False
    return not nome or re.search(rf"^(def|class) {nome}\b|^{nome} = ", arquivo.read_text(), re.M) is not None


def validar(issues: dict, pendencias: list | None = None) -> list[str]:
    """Erros bloqueiam. Artefato ausente em issue ainda sem PR de entrega é pendência (vai para `pendencias`)."""
    erros, vistos, testes = [], defaultdict(list), _testes_existentes()
    for num, issue in issues.items():
        ids = [r["id"] for r in issue["requisitos"]]
        for r in issue["requisitos"]:
            prefixo, _, n = r["id"].partition("-")
            if prefixo not in PREFIXOS or not re.fullmatch(r"\d{3}", n):
                erros.append(f"#{num}: id inválido {r['id']}")
            vistos[prefixo].append((int(n), num))
            for campo in ("texto", "implementacao", "verificacao"):
                if not r.get(campo):
                    erros.append(f"{r['id']}: campo `{campo}` vazio")
            for alvo in r.get("implementacao", []):
                if not _alvo_existe(alvo):
                    msg = f"{r['id']}: implementação não encontrada: {alvo}"
                    (erros if issue.get("pr") or pendencias is None else pendencias).append(msg)
            for v in r.get("verificacao", []):
                if v.startswith("tests::") and v.split("::")[1] not in testes:
                    erros.append(f"{r['id']}: teste não encontrado: {v}")
        for cen, req in (issue.get("cenarios") or {}).items():
            for rid in req:
                if rid not in ids:
                    erros.append(f"#{num} cenário {cen}: {rid} não pertence à issue")
    for prefixo, usados in vistos.items():
        nums = sorted(n for n, _ in usados)
        repetidos = {n for n in nums if nums.count(n) > 1}
        if repetidos:
            erros.append(f"{prefixo}: números repetidos {sorted(repetidos)}")
        if nums != list(range(1, len(nums) + 1)):
            erros.append(f"{prefixo}: sequência com lacunas ({nums[0]}–{nums[-1]}, {len(nums)} itens)")
    return erros


def _verificacao(v: str, testes: dict) -> str:
    if v.startswith("tests::"):
        nome = v.split("::")[1]
        return f"`{testes[nome]}::{nome}`"
    return v


def _entrega(pr) -> str:
    return f"#{pr}" if pr else "em andamento"


def gerar_matriz(issues: dict) -> str:
    testes = _testes_existentes()
    total = sum(len(i["requisitos"]) for i in issues.values())
    com_teste = sum(any(v.startswith("tests::") for v in r["verificacao"]) for i in issues.values() for r in i["requisitos"])
    entregues = sum(len(i["requisitos"]) for i in issues.values() if i.get("pr"))
    linhas = [
        "# Matriz de rastreabilidade dos requisitos",
        "",
        "Gerada por `scripts/rastreabilidade.py` a partir de `requisitos.yml` (fonte única). Não edite à mão.",
        "",
        f"**{total} itens** · {entregues} entregues · {total - entregues} em andamento · "
        f"**{com_teste} com teste automatizado** ({com_teste / total:.0%}); os demais são verificados por execução "
        "de notebook ou revisão.",
        "",
        "Prefixos: " + " · ".join(f"`{p}` {d}" for p, d in PREFIXOS.items()) + ".",
        "",
        "| ID | Requisito | Issue | Implementação | Verificação | Entrega |",
        "|---|---|---|---|---|---|",
    ]
    for num, issue in issues.items():
        for r in issue["requisitos"]:
            impl = "<br>".join(f"`{a}`" for a in r["implementacao"])
            verif = "<br>".join(_verificacao(v, testes) for v in r["verificacao"])
            linhas.append(f"| **{r['id']}** | {r['texto']} | #{num} | {impl} | {verif} | {_entrega(issue.get('pr'))} |")
    return "\n".join(linhas) + "\n"


def _corpo_requisitos(issue: dict, marcados: list[bool], testes: dict) -> str:
    blocos = []
    for i, r in enumerate(issue["requisitos"]):
        marca = "x" if (marcados[i] if i < len(marcados) else bool(issue.get("pr"))) else " "
        impl = ", ".join(f"`{a}`" for a in r["implementacao"])
        verif = ", ".join(_verificacao(v, testes) for v in r["verificacao"])
        blocos.append(f"- [{marca}] **{r['id']}** {r['texto']}\n"
                      f"  ↳ Implementação: {impl} · Verificação: {verif} · Entrega: {_entrega(issue.get('pr'))}")
    return "\n".join(blocos)


def reescrever_corpo(corpo: str, issue: dict) -> str:
    """Reescreve a lista de requisitos (preservando o que está marcado) e liga os cenários aos requisitos."""
    testes = _testes_existentes()
    for secao in SECOES:
        m = re.search(rf"({re.escape(secao)}\n\n)(.*?)(\n\n### |\Z)", corpo, re.S)
        if m:
            itens = re.findall(r"^- \[([ x])\]", m.group(2), re.M)
            novo = _corpo_requisitos(issue, [c == "x" for c in itens], testes)
            corpo = corpo[:m.start(2)] + novo + corpo[m.end(2):]
            break
    for cen, req in (issue.get("cenarios") or {}).items():
        corpo = re.sub(rf"^(\*\*Cenário {cen}:\*\* [^\n]*?)(?: · verifica [^\n]*)?$",
                       lambda mm: f"{mm.group(1)} · verifica {', '.join(req)}", corpo, flags=re.M)
    return corpo


def sincronizar(issues: dict) -> None:
    for num, issue in issues.items():
        corpo = subprocess.run(["gh", "issue", "view", str(num), "-R", REPO, "--json", "body", "--jq", ".body"],
                               check=True, capture_output=True, text=True).stdout.rstrip("\n")
        novo = reescrever_corpo(corpo, issue)
        if novo != corpo:
            subprocess.run(["gh", "issue", "edit", str(num), "-R", REPO, "--body", novo], check=True,
                           capture_output=True)
            print(f"#{num}: atualizada")
        else:
            print(f"#{num}: sem mudança")


def main() -> int:
    issues = carregar()
    pendencias = []
    erros = validar(issues, pendencias)
    if erros:
        print("Fonte única inválida:\n  " + "\n  ".join(erros))
        return 1
    if pendencias:
        print("Pendências de issues em andamento:\n  " + "\n  ".join(pendencias))
    (RAIZ / "REQUISITOS.md").write_text(gerar_matriz(issues))
    print(f"REQUISITOS.md gerado ({sum(len(i['requisitos']) for i in issues.values())} itens)")
    if "--sincronizar" in sys.argv:
        sincronizar(issues)
    return 0


if __name__ == "__main__":
    sys.exit(main())
