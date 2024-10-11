import os
import pandas as pd
import logging
from config import TREATMENT_CSV_FOLDER_PATHS

logger = logging.getLogger(__name__)

def validate_treated_data(file_path):
    """
    Valida os dados tratados para garantir que sigam as regras especificadas.
    
    Verificações:
    1. Verifica a presença das colunas corretas.
    2. Verifica a ausência de valores nulos nas colunas 'data', 'valor', 'series'.
    3. Verifica a continuidade da coluna 'data' (datas diárias sem gaps).
    4. Verifica se há valores duplicados.
    
    Retorna True se todos os critérios forem satisfeitos, caso contrário, False.
    """
    try:
        logger.info(f"Validando arquivo tratado: {file_path}")
        
        # Carregar o arquivo CSV tratado
        df = pd.read_csv(file_path)
        logger.info("Arquivo CSV carregado com sucesso.")

        # Verificar se as colunas estão presentes e na ordem correta
        expected_columns = ['data', 'valor', 'series']
        if list(df.columns) != expected_columns:
            logger.error(f"Colunas incorretas no arquivo. Esperado: {expected_columns}, Encontrado: {list(df.columns)}")
            return False
        
        # Verificar a existência de valores nulos
        if df.isnull().any().any():
            logger.error("Existem valores nulos no arquivo.")
            return False
        
        # Verificar continuidade das datas (sem gaps)
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        if df['data'].isnull().any():
            logger.error("Existem datas inválidas após a conversão para datetime.")
            return False

        # Garantir que as datas estão contínuas (sem gaps)
        date_diff = (df['data'].diff().dt.days.dropna() != 1).any()
        if date_diff:
            logger.error("Existem gaps nas datas. As datas não estão contínuas.")
            return False
        
        # Verificar duplicatas
        if df.duplicated().any():
            logger.error("Existem linhas duplicadas no arquivo.")
            return False
        
        # Se todas as validações forem bem-sucedidas
        logger.info(f"Validação concluída com sucesso para o arquivo: {file_path}")
        return True
    
    except Exception as e:
        logger.error(f"Erro ao validar o arquivo: {file_path} - {str(e)}")
        return False


def validate_all_treated_data():
    """
    Função que percorre todos os arquivos tratados e aplica a validação.
    """
    logger.info("Iniciando a validação de todos os arquivos tratados...")
    validation_results = {}
    
    for filename in os.listdir(TREATMENT_CSV_FOLDER_PATHS):
        if filename.endswith(".csv"):
            file_path = os.path.join(TREATMENT_CSV_FOLDER_PATHS, filename)
            is_valid = validate_treated_data(file_path)
            validation_results[filename] = is_valid
    
    logger.info("Validação concluída para todos os arquivos tratados.")
    return validation_results
