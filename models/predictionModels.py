import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt
import numpy as np


def verificar_valores_ausentes(data):
    """
    Verifica se o dataset contém valores ausentes ou infinitos.

    Args:
        data (pd.DataFrame): O DataFrame a ser verificado.

    Returns:
        bool: True se houver valores ausentes ou infinitos, caso contrário False.
    """
    # Conta valores ausentes em cada coluna
    missing = data.isnull().sum()
    # Conta valores infinitos em cada coluna
    infinite = data.isin([float('inf'), float('-inf')]).sum()

    # Exibe colunas com valores ausentes, se houver
    if missing.any():
        print(f"Valores ausentes encontrados:\n{missing[missing > 0]}")
    # Exibe colunas com valores infinitos, se houver
    if infinite.any():
        print(f"Valores infinitos encontrados:\n{infinite[infinite > 0]}")

    return missing.any() or infinite.any()


def tratar_valores_ausentes(data):
    """
    Trata valores ausentes e infinitos no dataset.

    Args:
        data (pd.DataFrame): O DataFrame a ser tratado.

    Returns:
        pd.DataFrame: DataFrame com valores ausentes ou infinitos tratados.
    """
    print("Interpolando valores ausentes e removendo valores infinitos...")
    # Interpola valores ausentes linearmente
    data = data.interpolate(method='linear', limit_direction='both')
    # Substitui valores infinitos por NaN
    data = data.replace([float('inf'), float('-inf')], pd.NA)
    # Remove valores NaN
    data = data.dropna()
    return data


def preparar_dados(pivoted_data, target_col, exog_vars):
    """
    Prepara os dados para treinamento e teste de modelos.

    Args:
        pivoted_data (pd.DataFrame): Dataset pivotado com colunas de variáveis.
        target_col (str): Nome da variável alvo.
        exog_vars (list): Lista de variáveis exógenas.

    Returns:
        Tuple: Dados de treino e teste para as variáveis alvo e exógenas.
    """
    if verificar_valores_ausentes(pivoted_data):
        pivoted_data = tratar_valores_ausentes(pivoted_data)

    # Garante que a frequência temporal seja consistente
    pivoted_data = pivoted_data.asfreq('D')

    # Define a variável alvo (y) e as exógenas (X)
    y = pivoted_data[target_col]
    X = pivoted_data[exog_vars] if exog_vars else None

    # Divide os dados entre treino (80%) e teste (20%)
    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (X[:train_size], X[train_size:]) if X is not None else (None, None)

    return y_train, y_test, X_train, X_test


def prever_linear_regression(y_train, y_test, X_train, X_test):
    """
    Realiza previsão usando Regressão Linear.

    Args:
        y_train, y_test: Dados de treino e teste da variável alvo.
        X_train, X_test: Dados de treino e teste das variáveis exógenas.

    Returns:
        Avaliação do modelo e previsões.
    """
    print("Executando Regressão Linear...")
    model = LinearRegression()
    model.fit(X_train, y_train)  # Treina o modelo
    y_pred = model.predict(X_test)  # Faz previsões
    return y_pred

def prever_ridge_regression(y_train, y_test, X_train, X_test):
    """
    Realiza previsão usando Ridge.

    Args:
        y_train, y_test: Dados de treino e teste da variável alvo.
        X_train, X_test: Dados de treino e teste das variáveis exógenas.

    Returns:
        Avaliação do modelo e previsões.
    """
    print("Executando Ridge...")
    
    alpha_grid = {'alpha': np.logspace(-4, 1, 50)}  
    ridge = GridSearchCV(Ridge(), param_grid=alpha_grid, cv=5, scoring='neg_mean_squared_error')
    ridge.fit(X_train, y_train)  # Fit Ridge with cross-validation
    best_ridge = ridge.best_estimator_  # Retrieve the best model

    y_pred = best_ridge.predict(X_test)  # Faz previsões
    return y_pred


def prever_lasso_regression(y_train, y_test, X_train, X_test):
    """
    Realiza previsão usando Lasso.

    Args:
        y_train, y_test: Dados de treino e teste da variável alvo.
        X_train, X_test: Dados de treino e teste das variáveis exógenas.

    Returns:
        Avaliação do modelo e previsões.
    """
    print("Executando Ridge...")
    alpha_grid = {'alpha': np.logspace(-4, 1, 50)}  
    lasso = GridSearchCV(Lasso(max_iter=10000), param_grid=alpha_grid, cv=5, scoring='neg_mean_squared_error')
    lasso.fit(X_train, y_train)  # Fit Lasso with cross-validation
    best_lasso = lasso.best_estimator_  # Retrieve the best model
    print(f"Best alpha for Lasso: {lasso.best_params_['alpha']}")
    y_pred = best_lasso.predict(X_test)  # Faz previsões
    return y_pred


def prever_random_forest(y_train, y_test, X_train, X_test):
    """
    Realiza previsão usando Random Forest.

    Args:
        y_train, y_test: Dados de treino e teste da variável alvo.
        X_train, X_test: Dados de treino e teste das variáveis exógenas.

    Returns:
        Avaliação do modelo e previsões.
    """
    print("Executando Random Forest...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)  # Treina o modelo
    y_pred = model.predict(X_test)  # Faz previsões
    return avaliar_modelo(y_test, y_pred, "Random Forest")


def verificar_estacionaridade(series):
    """
    Verifica se a série temporal é estacionária usando o teste ADF.

    Args:
        series (pd.Series): Série temporal a ser verificada.

    Returns:
        bool: True se for estacionária, caso contrário False.
    """
    adf_result = adfuller(series.dropna())  # Teste ADF
    print(f"ADF Statistic: {adf_result[0]}")
    print(f"p-value: {adf_result[1]}")
    return adf_result[1] <= 0.05  # Se p-value <= 0.05, a série é estacionária


def previsao_arima(y_train, y_test, order=(1, 1, 1)):
    """
    Realiza previsão usando o modelo ARIMA.

    Args:
        y_train, y_test: Dados de treino e teste da variável alvo.
        order (tuple): Parâmetros (p, d, q) do ARIMA.

    Returns:
        Avaliação do modelo e previsões.
    """
    print("Executando ARIMA...")
    try:
        model = ARIMA(y_train, order=order)  # Configura o modelo ARIMA
        results = model.fit()  # Ajusta o modelo
        y_pred = results.forecast(steps=len(y_test))  # Faz previsões
        return avaliar_modelo(y_test, y_pred, "ARIMA")
    except Exception as e:
        print(f"Erro durante a previsão ARIMA: {str(e)}")
        return None


def previsao_sarimax(y_train, y_test, X_train, X_test, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)):
    """
    Realiza previsão usando o modelo SARIMAX.

    Args:
        y_train, y_test: Dados de treino e teste da variável alvo.
        X_train, X_test: Dados de treino e teste das variáveis exógenas.
        order (tuple): Parâmetros (p, d, q) do SARIMAX.
        seasonal_order (tuple): Parâmetros sazonais (P, D, Q, m) do SARIMAX.

    Returns:
        Avaliação do modelo e previsões.
    """
    print("Executando SARIMAX...")
    try:
        model = SARIMAX(y_train, exog=X_train, order=order, seasonal_order=seasonal_order)
        results = model.fit(disp=False)  # Ajusta o modelo
        y_pred = results.forecast(steps=len(y_test), exog=X_test)  # Faz previsões
        return avaliar_modelo(y_test, y_pred, "SARIMAX")
    except Exception as e:
        print(f"Erro durante a previsão SARIMAX: {str(e)}")
        return None


def avaliar_modelo(y_test, y_pred, model_name):
    """
    Avalia o desempenho do modelo e exibe resultados.

    Args:
        y_test (pd.Series): Valores reais.
        y_pred (np.array): Valores previstos.
        model_name (str): Nome do modelo avaliado.

    Returns:
        tuple: Previsões, erro absoluto médio e RMSE.
    """
    mae = mean_absolute_error(y_test, y_pred)  # Calcula o MAE
    rmse = mean_squared_error(y_test, y_pred, squared=False)  # Calcula o RMSE

    print(f"Modelo: {model_name}")
    print(f"Erro Absoluto Médio (MAE): {mae}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse}")

    # Plota os resultados reais vs previstos
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label="Valores Reais (IPCA)", color="blue")
    plt.plot(y_test.index, y_pred, label=f"Previsões ({model_name})", color="orange")
    plt.title(f"Previsões do IPCA com {model_name}")
    plt.xlabel("Data")
    plt.ylabel("IPCA")
    plt.legend()
    plt.show()

    return y_pred, mae, rmse
