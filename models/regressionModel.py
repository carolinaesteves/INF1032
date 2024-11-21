import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

def verificar_valores_ausentes(data):
    """Verifica valores ausentes no dataset."""
    missing = data.isnull().sum()
    if missing.any():
        print(f"Valores ausentes encontrados:\n{missing[missing > 0]}")
        return True
    print("Nenhum valor ausente encontrado.")
    return False

def tratar_valores_ausentes(data):
    """Trata os valores ausentes interpolando linearmente."""
    print("Interpolando valores ausentes...")
    return data.interpolate(method='linear', limit_direction='both')

def previsao_ipca_linear(merged_data, target_col='ipca_bcb', exog_cols=None):
    """Faz a previsão do IPCA usando Regressão Linear."""
    try:
        if verificar_valores_ausentes(merged_data):
            merged_data = tratar_valores_ausentes(merged_data)

        X = merged_data[exog_cols]
        y = merged_data[target_col]

        train_size = int(len(y) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        linear_model = LinearRegression()
        linear_model.fit(X_train, y_train)

        y_pred = linear_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)

        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, label="Valores Reais (IPCA)", color="blue")
        plt.plot(y_test.index, y_pred, label="Previsões (Linear Regression)", color="orange")
        plt.title("Previsões do IPCA com Regressão Linear (Dados Interpolados)")
        plt.xlabel("Data")
        plt.ylabel("IPCA")
        plt.legend()
        plt.show()

        print(f"Erro Absoluto Médio (MAE): {mae}")
        print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse}")

        return y_pred, mae, rmse
    except Exception as e:
        print(f"Erro durante a previsão: {str(e)}")
        return None
