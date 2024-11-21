import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from config import TREATMENT_CSV_FOLDER_PATHS

# Função para carregar e combinar todas as séries tratadas
def load_combined_data(folder_path):
    all_data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)
            df['data'] = pd.to_datetime(df['data'])
            all_data.append(df)
    combined_data = pd.concat(all_data, axis=0)
    combined_data = combined_data.pivot(index='data', columns='series', values='valor').reset_index()
    return combined_data

# Função para exibir correlações com o IPCA
def analyze_correlation(combined_data):
    correlation_matrix = combined_data.corr()
    ipca_correlation = correlation_matrix['ipca_bcb'].sort_values(ascending=False)
    print("Correlação com o IPCA:\n", ipca_correlation)

# Função para rodar modelo SARIMAX
def run_sarimax(combined_data, exog_vars, target_var='ipca_bcb'):
    combined_data.fillna(method='ffill', inplace=True)
    
    exog_data = combined_data[exog_vars]
    target_data = combined_data[target_var]
    
    train_size = int(len(target_data) * 0.8)
    target_train, target_test = target_data[:train_size], target_data[train_size:]
    exog_train, exog_test = exog_data[:train_size], exog_data[train_size:]

    # Garantir que não há valores inválidos
    exog_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    exog_train.dropna(inplace=True)
    target_train = target_train.loc[exog_train.index]  # Alinhar índices
    if exog_train.isnull().values.any() or target_train.isnull().values.any():
        raise ValueError("Ainda há valores NaN ou inf após o tratamento.")

    # Modelo SARIMAX
    model = SARIMAX(target_train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()
    
    # Previsão
    forecast = results.forecast(steps=len(target_test), exog=exog_test)
    mae = mean_absolute_error(target_test, forecast)
    
    print(f"Previsões do IPCA:\n{forecast}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    return forecast, mae

# Função principal para executar todo o processo
def main():
    print("Carregando dados...")
    combined_data = load_combined_data(TREATMENT_CSV_FOLDER_PATHS)
    
    print("Analisando correlação com o IPCA...")
    analyze_correlation(combined_data)
    
    print("Executando modelo SARIMAX...")
    exog_vars = ['cambio_bcb', 'commodities_bcb_composto_em_real', 'selic_bcb']
    run_sarimax(combined_data, exog_vars)

if __name__ == "__main__":
    main()
