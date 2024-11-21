import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA
import warnings
from config import TREATMENT_CSV_FOLDER_PATHS, COLUMNS_DATASET

# Supressão de warnings do modelo ARIMA
warnings.filterwarnings("ignore")

# Função para carregar e alinhar os dados, utilizando o nome da série alvo para prever inflação
def load_and_align_data_with_series_target(target_series='ipca_bcb'):
    combined_df = pd.DataFrame()
    
    # Combina cada CSV em um único DataFrame, usando o valor de 'series' como nome da coluna
    for filename in os.listdir(TREATMENT_CSV_FOLDER_PATHS):
        if filename.endswith(".csv"):
            file_path = os.path.join(TREATMENT_CSV_FOLDER_PATHS, filename)
            df = pd.read_csv(file_path)
            
            # Converte a coluna 'data' para datetime e organiza pelo índice temporal
            df[COLUMNS_DATASET['data']] = pd.to_datetime(df[COLUMNS_DATASET['data']], errors='coerce')
            df.set_index(COLUMNS_DATASET['data'], inplace=True)
            df.sort_index(inplace=True)
            
            # Usa o valor de 'series' como nome da coluna para alinhamento
            series_name = df[COLUMNS_DATASET['series']].iloc[0]
            combined_df[series_name] = df[COLUMNS_DATASET['valor']]
    
    # Interpolação para lidar com valores ausentes
    combined_df.interpolate(method='linear', limit_direction='both', inplace=True)
    
    # Define o alvo e as features, onde IPCA é o alvo e as demais séries são features
    y = combined_df[target_series] if target_series in combined_df.columns else None
    X = combined_df.drop(columns=[target_series], errors='ignore')
    
    return X, y

# Função para adicionar variáveis temporais ao conjunto de dados
def add_time_features(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    return df

# Carrega e organiza os dados
X, y = load_and_align_data_with_series_target()

# Adiciona variáveis temporais às features
X_transformed = add_time_features(X.copy())

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, shuffle=False)

# MODELO 1: Regressão Linear
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
lin_reg_pred = lin_reg_model.predict(X_test)

# Avaliação Regressão Linear
lin_reg_mse = mean_squared_error(y_test, lin_reg_pred)
lin_reg_mae = mean_absolute_error(y_test, lin_reg_pred)
lin_reg_r2 = r2_score(y_test, lin_reg_pred)

# MODELO 2: Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

# Avaliação Random Forest
rf_mse = mean_squared_error(y_test, rf_pred)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)

# MODELO 3: ARIMA - with warnings handled
arima_model = ARIMA(y_train, order=(2,1,1))
arima_fit = arima_model.fit()
arima_pred = arima_fit.forecast(steps=len(y_test))

# Avaliação ARIMA
arima_mse = mean_squared_error(y_test, arima_pred)
arima_mae = mean_absolute_error(y_test, arima_pred)
arima_r2 = r2_score(y_test, arima_pred)

# Resultados Finais
print("Resultados dos Modelos:")
print("\nRegressão Linear:")
print(f"MSE: {lin_reg_mse:.4f}, MAE: {lin_reg_mae:.4f}, R²: {lin_reg_r2:.4f}")

print("\nRandom Forest:")
print(f"MSE: {rf_mse:.4f}, MAE: {rf_mae:.4f}, R²: {rf_r2:.4f}")

print("\nARIMA (2,1,1):")
print(f"MSE: {arima_mse:.4f}, MAE: {arima_mae:.4f}, R²: {arima_r2:.4f}")
