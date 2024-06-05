import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline

# 1. Carregamento do arquivo CSV
df = pd.read_csv('DadosBasicos.csv')

# 2. Análise inicial dos dados
print(df.info())
print(df.describe())
print(df.isnull().sum())

# 3. Imputação de valores ausentes usando IterativeImputer (Machine Learning)
imputer = IterativeImputer(estimator=RandomForestClassifier(), max_iter=10, random_state=42)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 4. Remoção de dados inconsistentes e redundantes
# Vamos considerar valores inconsistentes como valores muito distantes das médias
# E valores duplicados como redundantes

# Removendo duplicatas
df_imputed.drop_duplicates(inplace=True)
df_imputed.to_csv("arquivo tratado")
# Removendo dados inconsistentes (outliers)
from scipy import stats
z_scores = stats.zscore(df_imputed.select_dtypes(include=['float64', 'int64']))
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
df_clean = df_imputed[filtered_entries]

# 5. Conversão de atributos simbólicos para numéricos
label_encoders = {}
for column in df_clean.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_clean[column] = le.fit_transform(df_clean[column])
    label_encoders[column] = le

# 6. Transformação de atributos numéricos (padronização)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_clean), columns=df_clean.columns)

# 7. Redução de dimensionalidade com PCA
pca = PCA(n_components=0.95)  # Mantém 95% da variância
df_pca = pd.DataFrame(pca.fit_transform(df_scaled))

# 8. Balanceamento dos dados usando undersampling
X = df_pca.drop('Q00201', axis=1)  
y = df_pca['Q00201']

undersampler = RandomUnderSampler(random_state=42)
X_res, y_res = undersampler.fit_resample(X, y)

# Unindo os dados balanceados
df_final = pd.concat([X_res, y_res], axis=1)

# Salvando o dataframe final
df_final.to_csv('DadosBasicos_preprocessed.csv', index=False)

print("Pré-processamento concluído e dados salvos em 'DadosBasicos_preprocessed.csv'.")