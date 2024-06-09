
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def verificaOneHot(df,variaveisContinuas):
    resp=list(df.columns)
    resp.remove('C006')
    resp.remove('C008')
    for col in variaveisContinuas:
        resp.remove(col)
    return resp

df = pd.read_csv('SemOutlierSemNAN.csv')
variaveisContinuas = ['K04302', 'P00404', 'P00104']
variaveisOneHot = verificaOneHot(df, variaveisContinuas)

# Codifica o atributo sexo, C006
le = LabelEncoder()
df['C006'] = le.fit_transform(df['C006'])

# Aplicar OneHotEncoder às variáveis categóricas
onehotencoder = OneHotEncoder(sparse=False, drop='first')  # drop='first' to avoid the dummy variable trap
onehot_encoded = onehotencoder.fit_transform(df[variaveisOneHot])

# Criar DataFrame com os nomes das novas colunas codificadas
onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=[
    f"{col}_{category}" for col, categories in zip(variaveisOneHot, onehotencoder.categories_) 
    for category in categories[1:]  # Skipping the first category due to drop='first'
])

# Concatenar o DataFrame original com as variáveis codificadas
df = df.drop(variaveisOneHot, axis=1)
df = pd.concat([df.reset_index(drop=True), onehot_encoded_df.reset_index(drop=True)], axis=1)


# Normaliza dados contínuos
scaler = StandardScaler()
df[variaveisContinuas] = scaler.fit_transform(df[variaveisContinuas])

print(df.head())
df.to_csv("codificacaoETransformacaoDasColunas.csv")

