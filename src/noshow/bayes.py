"""Rede Bayesiana discreta: aprendizado de estrutura, Manto de Markov, CPTs e inferência sobre o alvo.

A estrutura é aprendida com o Hill Climbing do pgmpy. As CPTs e a inferência são implementadas aqui:
com todas as variáveis observadas, P(alvo | evidência) depende só do Manto de Markov do alvo, então a
predição é um produto de consultas a tabelas, vetorizado sobre todas as linhas.
"""

import logging
import warnings

import networkx as nx
import numpy as np
import pandas as pd

TARGET = "no_show"


def learn_structure(X: pd.DataFrame, target: str = TARGET, scoring: str = "bic-d", max_indegree: int = 3) -> nx.DiGraph:
    """Hill Climbing sobre dados discretos. O alvo não pode ser pai de nenhuma variável: a falta acontece
    depois que todas as variáveis do agendamento são conhecidas, então seu Manto de Markov são seus pais."""
    from pgmpy.estimators import ExpertKnowledge, HillClimbSearch

    proibidas = [(target, c) for c in X.columns if c != target]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        logging.getLogger("pgmpy").setLevel(logging.ERROR)
        dag = HillClimbSearch(X.astype(str)).estimate(
            scoring_method=scoring, max_indegree=max_indegree, show_progress=False,
            expert_knowledge=ExpertKnowledge(forbidden_edges=proibidas))
    grafo = nx.DiGraph(dag.edges())
    grafo.add_nodes_from(X.columns)
    return grafo


def markov_blanket(grafo: nx.DiGraph, node: str = TARGET) -> list[str]:
    """Pais, filhos e pais dos filhos."""
    pais = set(grafo.predecessors(node))
    filhos = set(grafo.successors(node))
    copais = {p for f in filhos for p in grafo.predecessors(f)}
    return sorted((pais | filhos | copais) - {node})


def naive_bayes_graph(features: list[str], target: str = TARGET) -> nx.DiGraph:
    grafo = nx.DiGraph([(target, f) for f in features])
    grafo.add_node(target)
    return grafo


class DiscreteBN:
    """Rede Bayesiana discreta com CPTs estimadas com a priori BDeu (Dirichlet uniforme de tamanho `ess`).

    `states` define o domínio de cada variável; estados que não aparecem no treino recebem só a
    pseudo-contagem da priori, nunca probabilidade zero.
    """

    def __init__(self, grafo: nx.DiGraph, target: str = TARGET, ess: float = 1.0):
        self.grafo = grafo
        self.target = target
        self.ess = ess

    def fit(self, X: pd.DataFrame, states: dict[str, list] | None = None):
        nodes = list(self.grafo.nodes)
        self.states_ = {n: list(states[n]) if states and n in states else sorted(X[n].astype(str).unique())
                        for n in nodes}
        codes = self._encode(X[nodes])
        self.cpts_ = {}
        for n in nodes:
            pais = sorted(self.grafo.predecessors(n))
            shape = [len(self.states_[p]) for p in pais] + [len(self.states_[n])]
            q = int(np.prod(shape[:-1])) if pais else 1
            contagem = np.zeros(int(np.prod(shape)))
            idx = np.ravel_multi_index([codes[c] for c in pais + [n]], shape)
            np.add.at(contagem, idx, 1)
            contagem = contagem.reshape(shape) + self.ess / (q * shape[-1])  # BDeu
            self.cpts_[n] = (pais, contagem / contagem.sum(axis=-1, keepdims=True))
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """P(alvo = cada estado | Manto de Markov), linhas × estados do alvo."""
        mb = markov_blanket(self.grafo, self.target)
        codes = self._encode(X[mb]) if mb else {}
        n_y = len(self.states_[self.target])
        logp = np.zeros((len(X), n_y))
        for k in range(n_y):
            codes[self.target] = np.full(len(X), k)
            for node in [self.target] + list(self.grafo.successors(self.target)):
                pais, tabela = self.cpts_[node]
                with np.errstate(divide="ignore"):  # só ocorre com ess = 0
                    logp[:, k] += np.log(tabela[tuple(codes[c] for c in pais + [node])])
        logp -= logp.max(axis=1, keepdims=True)
        p = np.exp(logp)
        return p / p.sum(axis=1, keepdims=True)

    def n_params(self, nodes=None) -> int:
        """Parâmetros livres: Σ (estados − 1) × combinações dos pais."""
        nodes = self.grafo.nodes if nodes is None else nodes
        return int(sum((t.shape[-1] - 1) * np.prod(t.shape[:-1]) for n, (_, t) in self.cpts_.items() if n in nodes))

    def cpt_frame(self, node: str | None = None) -> pd.DataFrame:
        """CPT de um nó como tabela (linhas = combinações dos pais, colunas = estados do nó)."""
        node = node or self.target
        pais, tabela = self.cpts_[node]
        if len(pais) > 1:
            index = pd.MultiIndex.from_product([self.states_[p] for p in pais], names=pais)
        else:
            index = pd.Index(self.states_[pais[0]], name=pais[0]) if pais else pd.Index(["—"])
        return pd.DataFrame(tabela.reshape(-1, tabela.shape[-1]), index=index, columns=self.states_[node])

    def _encode(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        codes = {}
        for col in X.columns:
            mapa = {s: i for i, s in enumerate(self.states_[col])}
            serie = X[col].astype(str).map(mapa)
            if serie.isna().any():
                desconhecidos = X[col][serie.isna()].unique()[:5]
                raise ValueError(f"'{col}' tem estados fora do domínio: {list(desconhecidos)}")
            codes[col] = serie.to_numpy(dtype=int)
        return codes
