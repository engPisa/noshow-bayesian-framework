"""Domínios categóricos e agrupamento de cidades aprendidos apenas no treino."""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

MISSING = "ausente"
UNKNOWN = "desconhecido"


class FoldCategories(BaseEstimator, TransformerMixin):
    """Prepara categorias sem consultar o teste.

    Para reagrupar cidades corretamente, forneça a coluna original `city` junto de
    `city_grp`. Ela é auxiliar e não entra como preditor. Bases antigas sem `city`
    continuam aceitas, mas não permitem desfazer um agrupamento global anterior.
    """

    def __init__(self, categorical, min_city=300):
        self.categorical = categorical
        self.min_city = min_city

    @staticmethod
    def _text(s):
        return s.astype("string").fillna(MISSING).astype(str)

    def fit(self, X, y=None):
        self.states_ = {}
        if "city_grp" in self.categorical:
            source = X["city"] if "city" in X else X["city_grp"]
            counts = self._text(source).value_counts()
            self.cities_ = set(counts[counts >= self.min_city].index) - {MISSING}
        for c in self.categorical:
            s = self._city(X) if c == "city_grp" else self._text(X[c])
            self.states_[c] = sorted(set(s) | {MISSING, UNKNOWN})
            if c == "city_grp":
                self.states_[c] = sorted(set(self.states_[c]) | {"outras"})
        return self

    def _city(self, X):
        s = self._text(X["city"] if "city" in X else X["city_grp"])
        return s.where(s.isin(self.cities_) | s.eq(MISSING), "outras")

    def transform(self, X):
        X = X.copy()
        for c in self.categorical:
            s = self._city(X) if c == "city_grp" else self._text(X[c])
            X[c] = s.where(s.isin(self.states_[c]), UNKNOWN)
        return X
