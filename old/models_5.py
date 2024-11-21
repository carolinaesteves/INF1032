import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX
from xgboost import XGBRegressor
import warnings
from config import TREATMENT_CSV_FOLDER_PATHS, COLUMNS_DATASET

warnings.filterwarnings("ignore")  # Ignorar warnings

# Função para carregar e alinhar os dados
def load_and_align_data_with_series_target(target_series='ipca_bcb'):
    combined_df = pd.DataFrame()
    for filename in os.listdir(TREATMENT_CSV_FOLDER_PATHS):
        if filename.endswith(".csv"):
            file_path = os.path.join(TREATMENT_CSV_FOLDER_PATHS, filename)
            df = pd.read_csv(file_path)
            df[COLUMNS_DATASET['data']] = pd.to_datetime(df[COLUMNS_DATASET['data']], errors='coerce')
            df.set_index(COLUMNS_DATASET['data'], inplace=True)
            series_name = df[COLUMNS_DATASET['series']].iloc[0]
            combined_df[series_name] = df[COLUMNS_DATASET['valor']]
    combined_df.interpolate(method='linear', limit_direction='both', inplace=True)
    y = combined_df[target_series] if target_series in combined_df.columns else None
    X = combined_df.drop(columns=[target_series], errors='ignore')
    return X, y

# Função para adicionar variáveis temporais e transformações
def add_time_features_and_transform(df):
    df['year'] = df.index.year
    df['month'] = df.index.month
    df['day_of_year'] = df.index.dayofyear
    df['week_of_year'] = df.index.isocalendar().week
    df['quarter'] = df.index.quarter
    
    # Para cada coluna de série, adiciona médias móveis e diferenciação
    for col in df.columns:
        df[f'{col}_month_avg'] = df.groupby('month')[col].transform('mean')
        df[f'{col}_rolling_mean'] = df[col].rolling(window=12).mean()
        df[f'{col}_diff_1'] = df[col].diff(1)  # Diferenciação de 1º ordem
        
    return df.fillna(method='bfill').fillna(method='ffill')  # Preenchimento para evitar valores nulos

# Carrega e organiza os dados
X, y = load_and_align_data_with_series_target()

# Adiciona variáveis temporais e transformações
X_transformed = add_time_features_and_transform(X.copy())

# Divisão dos dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, shuffle=False)

# MODELO 1: SARIMA para capturar sazonalidade
sarima_model = SARIMAX(y_train, order=(2, 1, 1), seasonal_order=(1, 1, 1, 12))
sarima_fit = sarima_model.fit(disp=False)
sarima_pred = sarima_fit.forecast(steps=len(y_test))

# Avaliação SARIMA
sarima_mse = mean_squared_error(y_test, sarima_pred)
sarima_mae = mean_absolute_error(y_test, sarima_pred)
sarima_r2 = r2_score(y_test, sarima_pred)

# MODELO 2: Regressão Linear com transformações
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, y_train)
lin_reg_pred = lin_reg_model.predict(X_test)

# Avaliação Regressão Linear
lin_reg_mse = mean_squared_error(y_test, lin_reg_pred)
lin_reg_mae = mean_absolute_error(y_test, lin_reg_pred)
lin_reg_r2 = r2_score(y_test, lin_reg_pred)

# MODELO 3: XGBoost para capturar não linearidades
xgb_model = XGBRegressor(n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_pred = xgb_model.predict(X_test)

# Avaliação XGBoost
xgb_mse = mean_squared_error(y_test, xgb_pred)
xgb_mae = mean_absolute_error(y_test, xgb_pred)
xgb_r2 = r2_score(y_test, xgb_pred)

# Resultados Finais
print("Resultados dos Modelos:")
print("\nSARIMA (2,1,1)(1,1,1,12):")
print(f"MSE: {sarima_mse:.4f}, MAE: {sarima_mae:.4f}, R²: {sarima_r2:.4f}")

print("\nRegressão Linear com Transformações:")
print(f"MSE: {lin_reg_mse:.4f}, MAE: {lin_reg_mae:.4f}, R²: {lin_reg_r2:.4f}")

print("\nXGBoost:")
print(f"MSE: {xgb_mse:.4f}, MAE: {xgb_mae:.4f}, R²: {xgb_r2:.4f}")