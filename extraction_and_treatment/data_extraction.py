import logging
from config import SERIES_IDS, IPEADATA_SERIES_CODES, BCB_ENDPOINT, IPEADATA_ENDPOINT, CSV_SAVE_PATHS, EXTRACTION_CSV_FOLDER_PATHS, YAHOO_FINANCE_TICKERS
from extraction_and_treatment.BCBDataExtractor import BCBDataExtractor
from extraction_and_treatment.IpeadataExtractor import IpeadataExtractor
from extraction_and_treatment.YahooFinanceExtractor import YahooFinanceExtractor
import os
from utils.utilities import create_directory_if_not_exists, extract_and_retry

logger = logging.getLogger(__name__)

def process_bcb_data(series_name, series_id):
    """Processa os dados do BCB para uma série específica."""
    try:
        bcb_extractor = BCBDataExtractor(series_id=series_id, series_name=series_name, base_url=BCB_ENDPOINT)
        bcb_data = bcb_extractor.get_processed_data()
        csv_filename = os.path.join(EXTRACTION_CSV_FOLDER_PATHS, CSV_SAVE_PATHS['bcb'].format(series_name))
        create_directory_if_not_exists(EXTRACTION_CSV_FOLDER_PATHS, logger)  # Garantir que o diretório exista
        bcb_data.to_csv(csv_filename, index=False)
        logger.info(f"Dados do BCB para {series_name} salvos em '{csv_filename}'")
    except Exception as e:
        logger.error(f"Erro ao processar dados do BCB para {series_name}: {str(e)}")

def process_ipeadata(series_name, series_code):
    """Processa os dados do Ipeadata para uma série específica."""
    try:
        ipeadata_extractor = IpeadataExtractor(series_code=series_code, base_url=IPEADATA_ENDPOINT)
        pib_data = ipeadata_extractor.get_processed_data()
        csv_filename = os.path.join(EXTRACTION_CSV_FOLDER_PATHS, CSV_SAVE_PATHS['ipeadata'].format(series_name))
        create_directory_if_not_exists(EXTRACTION_CSV_FOLDER_PATHS, logger)  # Garantir que o diretório exista
        pib_data.to_csv(csv_filename, index=False)
        logger.info(f"Ipeadata para {series_name} salvo em '{csv_filename}'")
    except Exception as e:
        logger.error(f"Erro ao processar dados do Ipeadata para {series_name}: {str(e)}")

def process_yahoo_finance_data(ticker, start_date, end_date):
    """Processa os dados do Yahoo Finance para um ticker específico."""
    try:
        yahoo_extractor = YahooFinanceExtractor(ticker=ticker, start_date=start_date, end_date=end_date)
        yahoo_data = yahoo_extractor.get_processed_data()
        csv_filename = os.path.join(EXTRACTION_CSV_FOLDER_PATHS, CSV_SAVE_PATHS['yahoo'].format(ticker))
        create_directory_if_not_exists(EXTRACTION_CSV_FOLDER_PATHS, logger)  # Garantir que o diretório exista
        yahoo_data.to_csv(csv_filename, index=False)
        logger.info(f"Dados do Yahoo Finance para {ticker} salvos em '{csv_filename}'")
    except Exception as e:
        logger.error(f"Erro ao processar dados do Yahoo Finance para {ticker}: {str(e)}")

def extraction():
    """Função principal para iniciar o processo de extração de dados."""
    logger.info("Iniciando o processo de extração de dados...")
    
    # Processar dados do BCB
    for series_name, series_id in SERIES_IDS.items():
        extract_and_retry(process_bcb_data, logger, series_name, series_id, max_retries=1)
    
    # Processar dados do Ipeadata
    for series_name, series_code in IPEADATA_SERIES_CODES.items():
        extract_and_retry(process_ipeadata, logger, series_name, series_code, max_retries=1)

    # Processar dados do Yahoo Finance
    for ticker, date_range in YAHOO_FINANCE_TICKERS.items():
        extract_and_retry(process_yahoo_finance_data, logger, ticker, date_range['start'], date_range['end'], max_retries=1)

    logger.info("Processo de extração de dados concluído.")
