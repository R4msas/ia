
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

def verificaOneHot(df,variaveisContinuas):
    resp=list(df.columns)
    resp.remove('C006')
    for col in variaveisContinuas:
        resp.remove(col)
    return resp

df = pd.read_csv('SemOutlierSemNAN.csv')
variaveisContinuas = ['K04302', 'P00404', 'P00104']
variaveisOneHot = verificaOneHot(df, variaveisContinuas)

# Codifica o atributo sexo, C006
le = LabelEncoder()
df['C006'] = le.fit_transform(df['C006'])

onehotencoder = OneHotEncoder(drop='first')
onehot_encoded = onehotencoder.fit_transform(df[variaveisOneHot]).toarray()
onehot_encoded_df = pd.DataFrame(onehot_encoded, columns=onehotencoder.get_feature_names_out(variaveisOneHot))
df = df.drop(variaveisOneHot, axis=1)
df = pd.concat([df, onehot_encoded_df], axis=1)
# Normaliza dados contínuos
scaler = StandardScaler()
df[variaveisContinuas] = scaler.fit_transform(df[variaveisContinuas])

print(df.head())
df.to_csv("codificacaoETransformacaoDasColunas.csv")

