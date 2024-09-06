from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import numpy as np


class GaussianMultiNomialClassifier(ClassifierMixin, BaseEstimator):
    categorical_features = 45

    def __init__(self, priors, var_smoothing, alpha, weights_decision):
        # Gaussian params
        self.priors = priors
        self.var_smoothing = var_smoothing

        # Multinomial params
        self.alpha = alpha

        # Pesos de decision siendo [Gaussian, Multinomial]
        self.weight_decision = weights_decision

        # Definición de modelos
        self.gaussian_nb = GaussianNB(
            priors=self.priors, var_smoothing=self.var_smoothing
        )
        self.multinomial_nb = MultinomialNB(alpha=self.alpha)

    def fit(self, X, y):
        # Compruebo que tengan el mismo tamaño
        X, y = check_X_y(X, y)
        # Guardo las clases
        self.classes_ = unique_labels(y)

        # Entreno con Gaussian todas las características menos las categoricas
        self.gaussian_nb.fit(X[:, : -self.categorical_features], y)

        # Entreno con MultiNomial todas las características categoricas
        self.multinomial_nb.fit(X[:, -self.categorical_features :], y)

        return self

    def predict(self, X):
        pred = self.predict_proba(X)

        # Función de activación lineal
        return np.argmax(pred, axis=1)

    def get_params(self, deep=True):
        return {
            "priors": self.priors,
            "var_smoothing": self.var_smoothing,
            "alpha": self.alpha,
            "weights_decision": self.weight_decision,
        }

    def set_params(self, **params):
        for parameter, value in params.items():
            setattr(self, parameter, value)
        return self

    def predict_proba(self, X):
        # Obtengo predicción con Gaussian todas las características menos las categoricas
        y_pred_gaussian = self.gaussian_nb.predict_proba(
            X[:, : -self.categorical_features]
        )

        # Obtengo predición con MultiNomial todas las características categoricas
        y_pred_multinomial = self.multinomial_nb.predict_proba(
            X[:, -self.categorical_features :]
        )

        # Obtengo predicciones combinadas
        pred = (y_pred_gaussian * self.weight_decision[0]) + (
            y_pred_multinomial * self.weight_decision[1]
        )

        self.pred_proba = pred

        return self.pred_proba
