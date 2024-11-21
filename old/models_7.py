import pandas as pd
import numpy as np
import os
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from config import TREATMENT_CSV_FOLDER_PATHS

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

def analyze_correlation(combined_data):
    correlation_matrix = combined_data.corr()
    ipca_correlation = correlation_matrix['ipca_bcb'].sort_values(ascending=False)
    print("Correlação com o IPCA:\n", ipca_correlation)

def run_sarimax(combined_data, exog_vars, target_var='ipca_bcb'):
    # Configurar índice e sincronizar as séries temporais
    combined_data.set_index('data', inplace=True)

    # Preencher valores ausentes nas variáveis exógenas
    combined_data[exog_vars] = combined_data[exog_vars].interpolate(method='linear', limit_direction='both')

    # Remover linhas com valores ausentes na variável alvo ou exógenas
    combined_data = combined_data.dropna(subset=[target_var] + exog_vars)

    exog_data = combined_data[exog_vars]
    target_data = combined_data[target_var]
    
    train_size = int(len(target_data) * 0.8)
    target_train, target_test = target_data[:train_size], target_data[train_size:]
    exog_train, exog_test = exog_data[:train_size], exog_data[train_size:]

    # Diagnóstico: verificar NaN após sincronização
    print("\nSoma de NaN em target_train e exog_train após sincronização:")
    print(f"target_train: {target_train.isnull().sum()}")
    print(f"exog_train:\n{exog_train.isnull().sum()}")

    # Validar se ainda existem valores ausentes
    if target_train.isnull().values.any():
        raise ValueError("target_train contém valores NaN.")
    if exog_train.isnull().values.any():
        raise ValueError("exog_train contém valores NaN.")

    # Modelo SARIMAX
    model = SARIMAX(target_train, exog=exog_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    results = model.fit()

    forecast = results.forecast(steps=len(target_test), exog=exog_test)
    mae = mean_absolute_error(target_test, forecast)
    rmse = mean_squared_error(target_test, forecast, squared=False)

    print(f"Previsões do IPCA:\n{forecast}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse}")

    plt.figure(figsize=(10, 6))
    plt.plot(target_test.index, target_test, label="Valores Reais (IPCA)", color="blue")
    plt.plot(target_test.index, forecast, label="Previsões (SARIMAX)", color="orange")
    plt.title("Previsões do IPCA com SARIMAX")
    plt.xlabel("Tempo")
    plt.ylabel("IPCA")
    plt.legend()
    plt.show()

    return forecast, mae, rmse

def main():
    print("Carregando dados...")
    combined_data = load_combined_data(TREATMENT_CSV_FOLDER_PATHS)
    
    print("Analisando correlação com o IPCA...")
    analyze_correlation(combined_data)
    
    print("Executando modelo SARIMAX...")
    exog_vars = ['core_cpi_bcb', 'fx_sell_bcb', 'cambio_bcb', 'selic_bcb']
    run_sarimax(combined_data, exog_vars)

if __name__ == "__main__":
    main()
