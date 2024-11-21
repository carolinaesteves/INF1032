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
    combined_data.set_index('data', inplace=True)
    return combined_data

def run_sarimax(combined_data, exog_vars, target_var='ipca_bcb'):
    combined_data[exog_vars] = combined_data[exog_vars].interpolate(method='linear', limit_direction='both')
    combined_data = combined_data.dropna(subset=[target_var] + exog_vars)

    # Separar treino e teste
    train_size = int(len(combined_data) * 0.8)
    train_data = combined_data.iloc[:train_size]
    test_data = combined_data.iloc[train_size:]

    target_train = train_data[target_var]
    target_test = test_data[target_var]
    exog_train = train_data[exog_vars]
    exog_test = test_data[exog_vars]

    print("Treinando o modelo SARIMAX...")
    model = SARIMAX(target_train, exog=exog_train, order=(2, 1, 2), seasonal_order=(1, 1, 0, 12))
    results = model.fit(disp=False)

    print("Fazendo previsões...")
    forecast = results.forecast(steps=len(target_test), exog=exog_test)

    mae = mean_absolute_error(target_test, forecast)
    rmse = mean_squared_error(target_test, forecast, squared=False)

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

    print("Executando modelo SARIMAX...")
    exog_vars = ['core_cpi_bcb', 'fx_sell_bcb', 'cambio_bcb', 'selic_bcb']
    run_sarimax(combined_data, exog_vars)

if __name__ == "__main__":
    main()
