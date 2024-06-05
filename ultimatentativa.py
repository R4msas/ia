import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler

# 1. Carregar os dados
df = pd.read_csv('DadosBasicos.csv')

# 2. Imputação de dados ausentes com KNN Imputer
imputer = KNNImputer(n_neighbors=5)
df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

# 3. Remoção de dados inconsistentes e redundantes
# Exemplo: Remover duplicatas
df_imputed.drop_duplicates(inplace=True)

# Exemplo: Remover valores inconsistentes - Aqui você precisa definir suas próprias regras de inconsistência
# df_imputed = df_imputed[df_imputed['coluna_exemplo'] > 0]

# 4. Conversão simbólica numérica
# Vamos supor que existem colunas categóricas que precisam ser convertidas
categorical_columns = df_imputed.select_dtypes(include=['object']).columns
df_encoded = pd.get_dummies(df_imputed, columns=categorical_columns, drop_first=True)

# 5. Transformação de atributos numéricos
# Normalização/Z-score scaling
scaler = StandardScaler()
numerical_columns = df_encoded.select_dtypes(include=[np.number]).columns
df_scaled = df_encoded.copy()
df_scaled[numerical_columns] = scaler.fit_transform(df_encoded[numerical_columns])

# 6. Redução de dimensionalidade com PCA
pca = PCA(n_components=0.95)  # Mantendo 95% da variância explicada
df_reduced = pca.fit_transform(df_scaled)
df_reduced.toCSV("arquivoDimensionado")
# 7. Balanceamento dos dados com undersampling
X = df_reduced
y = df_encoded['Q00201']  
# Undersampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X, y)

# Combinar X_res e y_res em um DataFrame final
df_final = pd.DataFrame(X_res)
df_final['Q00201'] = y_res

# Salvar o DataFrame preprocessado
df_final.to_csv('DadosBasicos_preprocessado.csv', index=False)

print("Pré-processamento concluído e arquivo salvo como 'DadosBasicos_preprocessado.csv'")
