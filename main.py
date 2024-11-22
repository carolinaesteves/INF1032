import logging
import pandas as pd
from extraction_and_treatment.data_extraction import extraction  # Função para extração de dados
from extraction_and_treatment.data_treatment import treatment  # Função para tratamento inicial de dados
from validation.validateTreatedData import validate_all_treated_data  # Função para validação dos dados tratados
from eda.eda import run_eda  # Função para análise exploratória dos dados (EDA)
from models.dataProcessing import processar_e_unificar_dados, gerar_dados_futuros, unificar_com_dados_futuros
from models.predictionModels import preparar_dados, prever_linear_regression, prever_random_forest, previsao_arima, previsao_sarimax
from utils.logging import setup_logging
from config import ANALYSIS_FOLDER_PATH, TREATMENT_CSV_FOLDER_PATHS


def main():
    # Configura o sistema de logging para registrar mensagens e eventos
    setup_logging(app_name="data_processing")

    logging.info("Começando a aplicação...")  # Marca o início do processo

    try:
        # **Processo 1: Extração de Dados**
        # Extrai dados brutos de fontes definidas (ex.: APIs, bancos de dados, arquivos).
        # Descomente a linha abaixo se precisar executar a extração:
        # extraction()

        # **Processo 2: Tratamento Inicial dos Dados**
        # Aplica tratamento básico aos dados extraídos, como limpeza e formatação.
        # Descomente a linha abaixo se precisar executar o tratamento:
        # treatment()

        # **Processo 3: Validação dos Dados Tratados**
        # Verifica a consistência e a qualidade dos dados tratados antes de prosseguir.
        # Descomente a linha abaixo se precisar validar os dados tratados:
        # validate_all_treated_data()

        # **Processo 4: Análise Exploratória de Dados (EDA)**
        # Realiza análises descritivas e cria visualizações para entender os dados.
        # Descomente a linha abaixo se precisar executar o EDA:
        # run_eda()

        # **Processo 5: Processamento e Unificação dos Dados**
        # Lê arquivos de entrada, organiza os dados em um único dataframe consolidado e salva o resultado.
        dados_unificados = processar_e_unificar_dados(TREATMENT_CSV_FOLDER_PATHS, ANALYSIS_FOLDER_PATH)

        if dados_unificados is not None:  # Garante que os dados foram processados corretamente
            # **Processo 6: Gerar Dados Futuros**
            # Gera previsões futuras com base na média móvel dos dados históricos.
            previsoes_futuras = gerar_dados_futuros(dados_unificados, dias_previsao=60, janela=30)

            # **Processo 7: Unificar Dados Reais com Dados Futuros**
            # Combina os dados reais e as previsões, criando uma base consolidada.
            dados_completos = unificar_com_dados_futuros(dados_unificados, previsoes_futuras, ANALYSIS_FOLDER_PATH)

            # **Processo 8: Preparar Dados para Modelagem**
            # Carrega os dados consolidados e organiza as variáveis para os modelos de previsão.
            dados_completos = pd.read_csv(f"{ANALYSIS_FOLDER_PATH}/dados_completos.csv")
            dados_completos["data"] = pd.to_datetime(dados_completos["data"])  # Converte a coluna 'data' para datetime
            # Reorganiza os dados para que as séries fiquem como colunas, indexadas por data.
            merged_data = dados_completos.pivot(index="data", columns="series", values="valor")

            # Define a variável alvo (IPCA) e as variáveis explicativas (exógenas).
            target_col = "ipca_bcb"
            exog_vars = ["selic_bcb", "cambio_bcb", "fx_sell_bcb", "core_cpi_bcb"]

            # Divide os dados em conjuntos de treino e teste para modelagem.
            y_train, y_test, X_train, X_test = preparar_dados(merged_data, target_col, exog_vars)

            # **Processo 9: Executar Modelos de Previsão**
            # Regressão Linear
            prever_linear_regression(y_train, y_test, X_train, X_test)
            # Random Forest
            prever_random_forest(y_train, y_test, X_train, X_test)
            # Modelo ARIMA
            previsao_arima(y_train, y_test, order=(2, 0, 2))
            # Modelo SARIMAX
            previsao_sarimax(y_train, y_test, X_train, X_test, order=(2, 0, 2), seasonal_order=(1, 1, 1, 12))

        logging.info("Aplicação executada com sucesso.")  # Indica que o processo foi concluído sem erros.
    except Exception as e:
        # Registra qualquer erro ocorrido durante a execução
        logging.error(f"Erro durante a execução da aplicação: {str(e)}")


if __name__ == "__main__":
    # Ponto de entrada do script
    main()
