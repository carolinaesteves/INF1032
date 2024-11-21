import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import itertools

def verificar_valores_ausentes(data):
    """Verifica valores ausentes no dataset."""
    missing = data.isnull().sum()
    if missing.any():
        print(f"Valores ausentes encontrados:\n{missing[missing > 0]}")
        return True
    print("Nenhum valor ausente encontrado.")
    return False

def tratar_valores_ausentes(data):
    """Trata valores ausentes interpolando ou preenchendo com médias."""
    print("Tratando valores ausentes...")
    return data.interpolate(method='linear', limit_direction='both')

def preparar_dados(merged_data, target_col, exog_vars):
    """Prepara os dados para modelagem."""
    y = merged_data[target_col]
    X = merged_data[exog_vars] if exog_vars else None

    train_size = int(len(y) * 0.8)
    y_train, y_test = y[:train_size], y[train_size:]
    X_train, X_test = (X[:train_size], X[train_size:]) if X is not None else (None, None)

    return y_train, y_test, X_train, X_test

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

    return mae, rmse

def testar_parametros_arima(y_train, y_test, p_values, d_values, q_values):
    """Testa combinações de parâmetros ARIMA."""
    best_mae = float('inf')
    best_params = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model = ARIMA(y_train, order=(p, d, q))
            results = model.fit()
            y_pred = results.forecast(steps=len(y_test))
            mae = mean_absolute_error(y_test, y_pred)

            if mae < best_mae:
                best_mae = mae
                best_params = (p, d, q)

            print(f"ARIMA({p}, {d}, {q}) - MAE: {mae}")
        except Exception as e:
            print(f"Erro em ARIMA({p}, {d}, {q}): {e}")

    print(f"Melhores parâmetros ARIMA: {best_params} com MAE: {best_mae}")
    return best_params

def main():
    # Carregar dados
    caminho_dados = "analysis/dados_unificados.csv"
    dados = pd.read_csv(caminho_dados)
    dados['data'] = pd.to_datetime(dados['data'])
    merged_data = dados.pivot(index='data', columns='series', values='valor')

    # Configurar variáveis alvo e exógenas
    target_col = 'ipca_bcb'
    exog_vars = ['selic_bcb', 'cambio_bcb', 'fx_sell_bcb', 'core_cpi_bcb']

    # Verificar e tratar valores ausentes
    if verificar_valores_ausentes(merged_data):
        merged_data = tratar_valores_ausentes(merged_data)

    # Preparar os dados
    y_train, y_test, X_train, X_test = preparar_dados(merged_data, target_col, exog_vars)

    # Testar parâmetros ARIMA
    p_values = [0, 1, 2]
    d_values = [0, 1]
    q_values = [0, 1, 2]
    best_arima_params = testar_parametros_arima(y_train, y_test, p_values, d_values, q_values)
    if best_arima_params:
        print(f"Executando ARIMA com os melhores parâmetros: {best_arima_params}")
        model_arima = ARIMA(y_train, order=best_arima_params)
        results_arima = model_arima.fit()
        y_pred_arima = results_arima.forecast(steps=len(y_test))
        avaliar_modelo(y_test, y_pred_arima, "ARIMA")

if __name__ == "__main__":
    main()
