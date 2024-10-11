import logging
from config import SERIES_IDS, IPEADATA_SERIES_CODES, BCB_ENDPOINT, IPEADATA_ENDPOINT, CSV_SAVE_PATHS, EXTRACTION_CSV_FOLDER_PATHS
from extraction_and_treatment.BCBDataExtractor import BCBDataExtractor
from extraction_and_treatment.IpeadataExtractor import IpeadataExtractor
import os
import time

logger = logging.getLogger(__name__)

def extract_and_retry(func, *args, max_retries=1, **kwargs):
    """Funcao auxiliar para tentar extrair novamente em caso de falha."""
    for attempt in range(max_retries + 1):
        try:
            func(*args, **kwargs)
            return True  # Sucesso
        except Exception as e:
            logger.error(f"Erro em {func.__name__} na tentativa {attempt + 1}: {str(e)}")
            if attempt < max_retries:
                logger.info(f"Tentando novamente {func.__name__} (tentativa {attempt + 1})...")
                time.sleep(2)  # Delay antes de tentar novamente
            else:
                logger.error(f"Falhou apos {max_retries + 1} tentativas.")
                return False

def process_bcb_data(series_name, series_id):
    """Processa os dados do BCB para uma serie especifica."""
    try:
        bcb_extractor = BCBDataExtractor(series_id=series_id, series_name=series_name, base_url=BCB_ENDPOINT)
        bcb_data = bcb_extractor.get_processed_data()
        csv_filename = os.path.join(EXTRACTION_CSV_FOLDER_PATHS, CSV_SAVE_PATHS['bcb'].format(series_name))
        bcb_data.to_csv(csv_filename, index=False)
        logger.info(f"Dados do BCB para {series_name} salvos em '{csv_filename}'")
    except Exception as e:
        logger.error(f"Erro ao processar dados do BCB para {series_name}: {str(e)}")

def process_ipeadata(series_name, series_code):
    """Processa os dados do Ipeadata para uma serie especifica."""
    try:
        ipeadata_extractor = IpeadataExtractor(series_code=series_code, base_url=IPEADATA_ENDPOINT)
        pib_data = ipeadata_extractor.get_processed_data()
        csv_filename = os.path.join(EXTRACTION_CSV_FOLDER_PATHS, CSV_SAVE_PATHS['ipeadata'].format(series_name))
        pib_data.to_csv(csv_filename, index=False)
        logger.info(f"Dados do Ipeadata para {series_name} salvos em '{csv_filename}'")
    except Exception as e:
        logger.error(f"Erro ao processar dados do Ipeadata para {series_name}: {str(e)}")

def extraction():
    """Função principal para iniciar a extracao de dados."""
    logger.info("Iniciando o processo de extracao de dados...")
    for series_name, series_id in SERIES_IDS.items():
        extract_and_retry(process_bcb_data, series_name, series_id, max_retries=1)
    for series_name, series_code in IPEADATA_SERIES_CODES.items():
        extract_and_retry(process_ipeadata, series_name, series_code, max_retries=1)
    logger.info("Processo de extracao de dados concluido.")
