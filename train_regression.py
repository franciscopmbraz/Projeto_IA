
# Importação das bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy import stats
from scipy.stats import t as t_dist
from sklearn.preprocessing import StandardScaler

# Configuração do matplotlib
warnings.filterwarnings('default')
plt.switch_backend('Agg')
plt.style.use('seaborn-v0_8')
sns.set_palette('husl')
plt.rcParams['figure.dpi'] = 100

pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

print('✓ Bibliotecas carregadas com sucesso!')

# Carregamento dos dados
df = pd.read_csv('processed_lisboa_porto_air_quality.csv', sep=';')

# Limpeza dos dados
df.columns = df.columns.str.lower()
df_clean = df.dropna(subset=['no2'])
print(f'Dataset: {df_clean.shape[0]} observações × {df_clean.shape[1]} variáveis')

#selecionar variaveis numéricas para análise

# converter datetime
df_clean['datetime'] = pd.to_datetime(df_clean['datetime'])

# criar variáveis temporais
df_clean['hour'] = df_clean['datetime'].dt.hour
df_clean['day'] = df_clean['datetime'].dt.month
df_clean['month'] = df_clean['datetime'].dt.day

y = df_clean['no2']

features = [
    'nox', 'co', 'pm10', 'pm2.5', 'o3', 'so2',
    'temperature_c', 'humidity_percent',
    'wind_speed_kmh', 'pressure_hpa',
    'precipitation_mm',
    'day', 'month', 'hour'
]

X = df_clean[features]

print(X.head())
# 9157 observações
print("Shape:", X.shape)

#12 features
print(f"Features: {X.shape[1]}")
## Verificar NaNs
print("NaNs antes:")
print(X.isnull().sum())
# Preencher NaNs com a média das colunas numéricas
X = X.fillna(X.mean(numeric_only=True))
# Verificar novamente
print("\nNaNs depois:")
print(X.isnull().sum())


# Dividir os dados em treino e teste. 20% para teste, 80% para treino.


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")


# ajustar escala dos dados

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# treinar modelo

model = LinearRegression()
model.fit(X_train_scaled, y_train)

print("✓ Modelo treinado")


# Valores dados para avaliação

y_pred = model.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n Resultados:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R2: {r2:.2f}")

# salvar modelo e scaler  
# ainda nao esta a funcionar, verificar depois

import pickle

with open("regression_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Modelo e scaler guardados")

# salvar métricas em CSV

metrics = pd.DataFrame({
    "model": ["Linear Regression"],
    "rmse": [rmse],
    "mae": [mae],
    "r2": [r2]
})

metrics.to_csv("metrics_regression.csv", index=False)

print("Métricas guardadas")


