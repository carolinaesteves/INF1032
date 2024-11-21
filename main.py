import logging
from extraction_and_treatment.data_extraction import extraction
from extraction_and_treatment.data_treatment import treatment
from utils.logging import setup_logging
from validation.validateTreatedData import validate_all_treated_data
from models.models import run_models
from eda.eda import run_eda
from models.dataProcessing import processar_e_unificar_dados
from models.predictionModels import *
import pandas as pd
from config import ANALYSIS_FOLDER_PATH, TREATMENT_CSV_FOLDER_PATHS

def main():
    # Setup logging
    setup_logging(app_name='data_processing')

    logging.info("Comecando Aplicacao ...")

    try:
        # Start the extraction process
        #extraction()

        # Start the treatment process
        #treatment()

        #validate_all_treated_data()
        
        # Run EDA
        #run_eda()  # Chame a função de EDA

        # model 1 - Preencher com primeiro modelo ou validação dos dados
        #run_models()

      # Processamento e unificação dos dados
        dados_unificados = processar_e_unificar_dados(TREATMENT_CSV_FOLDER_PATHS, ANALYSIS_FOLDER_PATH)

        if dados_unificados is not None:
            # Carregar a base unificada
            dados_unificados = pd.read_csv(f"{ANALYSIS_FOLDER_PATH}/dados_unificados.csv")
            dados_unificados['data'] = pd.to_datetime(dados_unificados['data'])
            merged_data = dados_unificados.pivot(index='data', columns='series', values='valor')

            # Configurar variáveis alvo e exógenas
            target_col = 'ipca_bcb'
            exog_vars = ['selic_bcb', 'cambio_bcb', 'fx_sell_bcb', 'core_cpi_bcb']

            # Preparar os dados para modelagem
            y_train, y_test, X_train, X_test = preparar_dados(merged_data, target_col, exog_vars)

            # Executar modelos
            logging.info("Executando modelos de previsão...")

            # Regressão Linear
            prever_linear_regression(y_train, y_test, X_train, X_test)

            # Random Forest
            prever_random_forest(y_train, y_test, X_train, X_test)

            previsao_arima(y_train, y_test, order=(2, 0, 2))
            previsao_sarimax(y_train, y_test, X_train, X_test, order=(2, 0, 2), seasonal_order=(1, 1, 1, 12))


        logging.info("Aplicacao rodou com Sucesso.")
    except Exception as e:
        logging.error(f"Aplicacao falhou com Erro: {str(e)}")

if __name__ == "__main__":
    main()
