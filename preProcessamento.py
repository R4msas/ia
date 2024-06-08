import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
# from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from scipy.stats import zscore

# Definir o random state
RANDOM_STATE = 42
def defineOutliersIndex(nomeColuna, df):
    coluna = df[nomeColuna]
    Q1 = coluna.quantile(0.25)
    Q3 = coluna.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Filtra o DataFrame para manter apenas os valores dentro dos limites
    outliers_indices = df[(coluna < lower_bound) | (coluna > upper_bound)].index
    
    return outliers_indices


# def remove_outliers_iqr(data, factor=2):
#     Q1 = data.quantile(0.25)
#     Q3 = data.quantile(0.75)
#     IQR = Q3 - Q1
#     is_not_outlier = ~((data < (Q1 - factor * IQR)) | (data > (Q3 + factor * IQR))).any(axis=1)
#     return data[is_not_outlier]


def sumDrinks(value1, value2):
    value1 = 0 if pd.isna(value1) else value1
    value2 = 0 if pd.isna(value2) else value2
    
    total = value1 + value2

    return total if total <= 7 else 7

# Carregar os dados
df = pd.read_csv('Dados.csv')
print(df)

# Remover amostras onde não haja resposta numérica na classe "Q00301", diabetes
# df = df[pd.to_numeric(df['Q00301'], errors='coerce').notnull()]
variaveisContinuas=['K04302','P00404','P00104']

for col in variaveisContinuas:
    index = defineOutliersIndex(col, df)
    df = df.drop(index)

# df = df[df['K04302'] < 90]
# df = df[df['P00404'] < 300]

juice = df['P02002']
soda = df['P02001']
jointDrinks = [sumDrinks(value1, value2) for value1, value2 in zip(juice, soda)]

df = df.drop(columns=['P02002','P02001'])
df['Drinks'] = jointDrinks

print(df)
