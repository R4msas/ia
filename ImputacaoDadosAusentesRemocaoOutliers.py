import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from imblearn.over_sampling import SMOTE
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

def removeOutliers(df, variaveisContinuas):
    for col in variaveisContinuas:
        index = defineOutliersIndex(col, df)
        df = df.drop(index)
    return df

def verificaClassificatorias(df,variaveisContinuas):
    resp=list(df.columns)
    resp.remove('C008')#retira a idade do morador, pois não há tratamento a ser feito, já foram retirados os valores nulos
    for col in variaveisContinuas:
        resp.remove(col)
    return resp
    
# def imputarVariaveisContinuas(df, variaveisContinuas):
#     for col in variaveisContinuas:
#         if df[col].isnull().any():
#             not_null_idx = df[col].notnull()
#             model = HistGradientBoostingRegressor(random_state=RANDOM_STATE)
#             model.fit(df[not_null_idx], df[col][not_null_idx])
#             df.loc[~not_null_idx, col] = model.predict(df[~not_null_idx])
#     return df
def imputarVariaveisContinuas(df):
    imputer = KNNImputer(n_neighbors=5)
    df=pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
    return df

def imputarVariaveisClassificatorias(df, variaveisClassificatorias):
    for col in variaveisClassificatorias:
        if df[col].isnull().any():
            not_null_idx = df[col].notnull()
            model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
            model.fit(df[not_null_idx], df[col][not_null_idx])
            df.loc[~not_null_idx, col] = model.predict(df[~not_null_idx])
    return df

df = pd.read_csv('Dados.csv')

variaveisContinuas=['K04302','P00404','P00104']
variaveisClassificatorias=verificaClassificatorias(df,variaveisContinuas)
print(df.shape)
df=removeOutliers(df,variaveisContinuas)
print(df)
# df=imputarVariaveisContinuas(df, variaveisContinuas)
# print(df)
df=imputarVariaveisClassificatorias(df, variaveisClassificatorias)
df_imputed=imputarVariaveisContinuas(df)
arredondamento=pd.Series([0],index=['K04302'])
df_imputed.round(arredondamento)
print(df_imputed)
df_imputed.drop_duplicates(inplace=True)
df_imputed.to_csv("SemOutlierSemNAN.csv")
