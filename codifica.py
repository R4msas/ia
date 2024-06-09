import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

base = pd.read_csv("SemOutlierSemNAN.csv", sep=',')



from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ['K04302', 'P00404', 'P00104'] E C006 C008

# hot_enconder_C009 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_J001 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_J00101 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_K031 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_K04301 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_K04302 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_N001 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_N00101= OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_N010 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_N014 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_P02002 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_P02001 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_P027 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")
# hot_enconder_P50 = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform="pandas")

def verificaOneHot(df, variaveisContinuas):
    resp=list(df.columns)
    resp.remove('C006')
    resp.remove('Q00201')
    for col in variaveisContinuas:
        resp.remove(col)
    return resp

def processaOneHot(col: str, df: pd):
    hot_enconder_atual = OneHotEncoder(handle_unknown='ignore', sparse_output=False).set_output(transform='pandas')
    transform_atual = hot_enconder_atual.fit_transform(df[[col]])
    df = pd.concat([df, transform_atual], axis=1).drop(columns= [col])
    return df




df = pd.read_csv('SemOutlierSemNAN.csv')

print(df)

variaveisContinuas = ['K04302', 'P00404', 'P00104', 'C008']
variaveisOneHot = verificaOneHot(df, variaveisContinuas)
variaveisOneHot.pop(0)

print(variaveisOneHot)


for i in variaveisOneHot:
    df = processaOneHot(i, df)



scaler = StandardScaler()
df[variaveisContinuas] = scaler.fit_transform(df[variaveisContinuas])


df = df[[col for col in df.columns if col != 'Q00201'] + ['Q00201']]

print(df)
print(df.shape)

X_prev = df.iloc[: , 1:94].values
y_classe = df.iloc[: , 94].values


from sklearn.model_selection import train_test_split
X_treino, X_teste, y_treino, y_teste = train_test_split(X_prev, y_classe, test_size= 0.20, random_state= 42)


from imblearn.under_sampling import RandomUnderSampler

undersampler = RandomUnderSampler(random_state=42)

X_treino_bal, y_treino_bal = undersampler.fit_resample(X_treino, y_treino)

print("SHAPES")

print(X_treino_bal.shape)
print(X_teste.shape)

print(df.iloc[: , 1:94])

import pickle
with open("baseHiper.pkl", mode="wb") as f:
    pickle.dump([X_treino_bal, X_teste, y_treino_bal, y_teste], f)