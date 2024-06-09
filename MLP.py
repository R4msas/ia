import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.neural_network import MLPClassifier

# importa pickle
import pickle 
with open("base.pkl", "rb") as f:
        X_treino, X_teste, y_treino, y_teste = pickle.load(f)


# modelo = MLPClassifier()
# modelo.fit(X_treino, y_treino)


# modelo = MLPClassifier(max_iter=1000, verbose=True)
# modelo.fit(X_treino, y_treino)

from sklearn.model_selection import RandomizedSearchCV

clf = MLPClassifier(verbose=True, max_iter= 3000)
RandomParameters = {
    'solver': ['sgd', 'adam', 'lbfgs'],
    'activation': ['relu', 'tanh'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']

}

modelo = RandomizedSearchCV(clf, param_distributions= RandomParameters, n_jobs=-1 , n_iter= 100, cv= 2)
modelo.fit(X_treino, y_treino)

param = modelo.best_params_
score = modelo.best_score_
estimator = modelo.best_estimator_


print("Resultados")
print(param)
print(score)
print(estimator)

# cm = ConfusionMatrix(modelo)
# cm.fit(X_treino, y_treino)
# cm.score(X_teste, y_teste)



# previsoes = modelo.predict(X_teste)

# from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# print(accuracy_score(y_teste, previsoes))

# print(confusion_matrix(y_teste, previsoes))

# print(classification_report(y_teste, previsoes))