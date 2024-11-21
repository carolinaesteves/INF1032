import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def verificar_valores_ausentes(data):
    """Verifica valores ausentes ou infinitos no dataset."""
    missing = data.isnull().sum()
    infinite = data.isin([float('inf'), float('-inf')]).sum()

    if missing.any():
        print(f"Valores ausentes encontrados:\n{missing[missing > 0]}")
    if infinite.any():
        print(f"Valores infinitos encontrados:\n{infinite[infinite > 0]}")

    if missing.any() or infinite.any():
        return True

    print("Nenhum valor ausente ou infinito encontrado.")
    return False

def tratar_valores_ausentes(data):
    """Trata valores ausentes e infinitos."""
    print("Interpolando valores ausentes e removendo valores infinitos...")
    data = data.interpolate(method='linear', limit_direction='both')
    data = data.replace([float('inf'), float('-inf')], pd.NA)
    data = data.dropna()
    return data

def preparar_dados(pivoted_data, target_col, exog_vars):
    """Prepara os dados para modelagem."""
    if verificar_valores_ausentes(pivoted_data):
        pivoted_data = tratar_valores_ausentes(pivoted_data)

    # Verificar se a frequência é consistente
    pivoted_data = pivoted_data.asfreq('D')

    # Separar variáveis alvo e exógenas
    y = pivoted_data[target_col]
    X = pivoted_data[exog_vars] if exog_vars else None

    # Dividir entre treino e teste
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (X[:train_size], X[train_size:]) if X is not None else (None, None)

    return y_train, y_test, X_train, X_test

def prever_linear_regression(y_train, y_test, X_train, X_test):
    """Previsão com Regressão Linear."""
    print("Executando Regressão Linear...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return avaliar_modelo(y_test, y_pred, "Linear Regression")

def prever_random_forest(y_train, y_test, X_train, X_test):
    """Previsão com Random Forest."""
    print("Executando Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return avaliar_modelo(y_test, y_pred, "Random Forest")

def verificar_estacionaridade(series):
    """Verifica a estacionaridade da série temporal usando o teste ADF."""
    adf_result = adfuller(series.dropna())
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    if adf_result[1] <= 0.05:
        print("A série é estacionária (p-valor <= 0.05).")
        return True
    else:
        print("A série não é estacionária (p-valor > 0.05).")
        return False

def previsao_arima(y_train, y_test, order=(1, 1, 1)):
    """Previsão do IPCA com ARIMA."""
    print("Executando ARIMA...")
    try:
        # Treinar modelo ARIMA com parâmetros fixos
        model = ARIMA(y_train, order=order)
        results = model.fit()

        # Fazer previsões
        y_pred = results.forecast(steps=len(y_test))

        return avaliar_modelo(y_test, y_pred, "ARIMA")
    except Exception as e:
        print(f"Erro durante a previsão ARIMA: {str(e)}")
        return None

def previsao_sarimax(y_train, y_test, X_train, X_test, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
    """Previsão do IPCA com SARIMAX."""
    print("Executando SARIMAX...")
    try:
        # Treinar modelo SARIMAX com parâmetros fixos
        model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)

        # Fazer previsões
        y_pred = results.forecast(steps=len(y_test), exog=X_test)

        return avaliar_modelo(y_test, y_pred, "SARIMAX")
    except Exception as e:
        print(f"Erro durante a previsão SARIMAX: {str(e)}")
        return None

def avaliar_modelo(y_test, y_pred, model_name):
    """Avalia o modelo e exibe os resultados."""
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    print(f"Modelo: {model_name}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse}")

    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Valores Reais (IPCA)", color="blue")
    plt.plot(y_test.index, y_pred, label=f"Previsões ({model_name})", color="orange")
    plt.title(f"Previsões do IPCA com {model_name}")
    plt.xlabel("Data")
    plt.ylabel("IPCA")
    plt.legend()
    plt.show()

    return y_pred, mae, rmse

