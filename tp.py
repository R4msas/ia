import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from scipy.stats import zscore

def remove_outliers_iqr(data, factor=2):
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    is_not_outlier = ~((data < (Q1 - factor * IQR)) | (data > (Q3 + factor * IQR))).any(axis=1)
    return data[is_not_outlier]


# Definir o random state
RANDOM_STATE = 42

# Carregar os dados
df = pd.read_csv('DadosBasicos.csv')

# Remover amostras onde não haja resposta numérica na classe "Q00301"
df = df[pd.to_numeric(df['Q00301'], errors='coerce').notnull()]

# Separar features e target
X = df.drop('Q00301', axis=1)
y = df['Q00301']
print(f'X antes do tratamento: {X}')
X = remove_outliers_iqr(X, factor=2)
print(f'X depois do tratamento: {X}')

# Imputação de dados ausentes com HistGradientBoostingClassifier
# HistGradientBoostingClassifier lida nativamente com NaNs, portanto, não precisamos de IterativeImputer
# Treinar o HistGradientBoostingClassifier para cada coluna com valores ausentes
for column in X.columns:
    if X[column].isnull().any():
        not_null_idx = X[column].notnull()
        model = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
        model.fit(X[not_null_idx], X[column][not_null_idx])
        X.loc[~not_null_idx, column] = model.predict(X[~not_null_idx])

# Verificar se há dados restantes após a imputação
if X.isnull().any().any():
    raise ValueError("A imputação de dados ausentes falhou. Ainda há NaNs presentes no conjunto de dados.")

# Verificar se há dados restantes após a remoção de outliers
if X.empty:
    raise ValueError("O conjunto de dados ficou vazio após remover outliers.")

# Balanceamento dos dados (considerando um problema de classificação)
smote = SMOTE(sampling_strategy='auto', random_state=RANDOM_STATE)
X_res, y_res = smote.fit_resample(X, y)

# Combinar novamente o dataframe balanceado
df_res = pd.concat([pd.DataFrame(X_res, columns=X.columns), pd.DataFrame(y_res, columns=['Q00301'])], axis=1)

# Remover duplicatas
df_res = df_res.drop_duplicates()

# Verificar se há dados restantes após a remoção de duplicatas
if df_res.empty:
    raise ValueError("O conjunto de dados ficou vazio após remover duplicatas.")

# Remoção de colunas redundantes feita manualmente
# Exemplo de remoção: df_res = df_res.drop(['redundant_column'], axis=1)

# Conversão simbólica para numérica
label_encoders = {}
for column in df_res.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_res[column] = le.fit_transform(df_res[column])
    label_encoders[column] = le

# Transformação de atributos numéricos
scaler = StandardScaler()
df_res[df_res.columns] = scaler.fit_transform(df_res[df_res.columns])

# Redução de dimensionalidade
# Aqui, usamos PCA como exemplo
pca = PCA(n_components=0.95, random_state=RANDOM_STATE)  # Manter 95% da variância
df_res_reduced = pd.DataFrame(pca.fit_transform(df_res), columns=[f'PC{i}' for i in range(1, pca.n_components_+1)])

# Garantir que os tamanhos de df_res_reduced e y_res sejam consistentes
if len(df_res_reduced) != len(y_res):
    min_len = min(len(df_res_reduced), len(y_res))
    df_res_reduced = df_res_reduced[:min_len]
    y_res = y_res[:min_len]

# Método de amostragem
# Aqui, dividimos os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(df_res_reduced, y_res, test_size=0.2, random_state=RANDOM_STATE)

# Verificação final
print(f'Forma dos dados de treinamento: {X_train.shape}')
print(f'Forma dos dados de teste: {X_test.shape}')
