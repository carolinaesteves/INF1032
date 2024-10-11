import os
import pandas as pd
import logging
from config import EXTRACTION_CSV_FOLDER_PATHS, TREATMENT_CSV_FOLDER_PATHS
from utils.utilities import create_directory_if_not_exists

logger = logging.getLogger(__name__)



def treat_ipeadata_file(file_path, output_folder):
    try:
        logger.info(f"Processando arquivo Ipeadata: {file_path}")
        
        # Carregar o arquivo CSV
        df = pd.read_csv(file_path)
        logger.info(f"Arquivo CSV carregado com sucesso: {file_path}")
        
        # Verificar e renomear colunas
        if 'SERCODIGO' in df.columns and 'Date' in df.columns and 'Value' in df.columns:
            df = df[['Date', 'SERCODIGO', 'Value']]
            df.rename(columns={'Date': 'data', 'SERCODIGO': 'series', 'Value': 'valor'}, inplace=True)
        else:
            raise ValueError("O arquivo CSV não contém as colunas esperadas: 'SERCODIGO', 'Date', 'Value'.")
        
        # Ordenar as colunas
        df = df[['data', 'valor', 'series']]

        # Converter a coluna 'data' para datetime
        df['data'] = pd.to_datetime(df['data'], errors='coerce').dt.normalize()
        
        # Verificar valor único na coluna 'series'
        series_value = df['series'].dropna().unique()
        if len(series_value) != 1:
            raise ValueError("Valores múltiplos inesperados na coluna 'series'.")
        series_value = series_value[0]

        # Definir 'data' como índice e reindexar para preencher lacunas com datas diárias
        df.set_index('data', inplace=True)
        df_reindexed = df.resample('D').asfreq()
        
        # Interpolar valores faltantes
        df_reindexed['valor'] = df_reindexed['valor'].combine_first(df['valor'])
        df_reindexed['valor'] = df_reindexed['valor'].interpolate(method='linear', limit_direction='both')
        
        # Preencher a coluna 'series'
        df_reindexed['series'] = series_value
        
        # Resetar índice e converter 'data' para string
        df_reindexed.reset_index(inplace=True)
        df_reindexed['data'] = df_reindexed['data'].dt.strftime('%Y-%m-%d')

        # Garantir que o diretório de saída exista
        create_directory_if_not_exists(output_folder, logger)
        
        # Salvar o arquivo tratado
        treated_file_path = os.path.join(output_folder, os.path.basename(file_path))
        df_reindexed.to_csv(treated_file_path, index=False)
        
        logger.info(f"Arquivo tratado salvo com sucesso: {treated_file_path}")
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo Ipeadata: {file_path} - {str(e)}")

def treat_bcbdata_file(file_path, output_folder):
    try:
        logger.info(f"Processando arquivo BCB: {file_path}")
        
        # Carregar o arquivo CSV
        df = pd.read_csv(file_path)
        
        # Verificar e tratar colunas
        expected_columns = ['data', 'valor', 'series']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltando colunas: {missing_columns}")

        # Converter 'data' para datetime
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        
        # Definir 'data' como índice e interpolar valores
        df.set_index('data', inplace=True)
        df = df.resample('D').asfreq()
        df['valor'] = df['valor'].interpolate(method='linear')

        # Resetar índice e salvar
        df.reset_index(inplace=True)
        df['data'] = df['data'].dt.strftime('%Y-%m-%d')
        
        # Preencher e salvar
        df['series'] = df['series'].ffill().bfill()
        create_directory_if_not_exists(output_folder, logger)
        treated_file_path = os.path.join(output_folder, os.path.basename(file_path))
        df.to_csv(treated_file_path, index=False)

        logger.info(f"Arquivo BCB tratado salvo com sucesso: {treated_file_path}")
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo BCB: {file_path} - {str(e)}")

def treat_yahoo_finance_file(file_path, output_folder):
    try:
        logger.info(f"Processando arquivo Yahoo Finance: {file_path}")
        
        # Carregar e tratar o arquivo CSV
        df = pd.read_csv(file_path)
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        df.set_index('data', inplace=True)
        df = df.resample('D').asfreq()
        df['valor'] = df['valor'].interpolate(method='linear')
        df.reset_index(inplace=True)
        df['data'] = df['data'].dt.strftime('%Y-%m-%d')
        df['series'] = df['series'].ffill().bfill()

        create_directory_if_not_exists(output_folder, logger)
        treated_file_path = os.path.join(output_folder, os.path.basename(file_path))
        df.to_csv(treated_file_path, index=False)

        logger.info(f"Arquivo Yahoo Finance tratado salvo com sucesso: {treated_file_path}")
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo Yahoo Finance: {file_path} - {str(e)}")

def treatment():
    logger.info("Iniciando o processo de tratamento de arquivos...")
    for filename in os.listdir(EXTRACTION_CSV_FOLDER_PATHS):
        if filename.endswith(".csv"):
            file_path = os.path.join(EXTRACTION_CSV_FOLDER_PATHS, filename)
            logger.info(f"Processando arquivo: {file_path}")
            if "ipeadata" in filename.lower():
                treat_ipeadata_file(file_path, TREATMENT_CSV_FOLDER_PATHS)
            elif "bcb" in filename.lower():
                treat_bcbdata_file(file_path, TREATMENT_CSV_FOLDER_PATHS)
            elif "yahoo" in filename.lower():
                treat_yahoo_finance_file(file_path, TREATMENT_CSV_FOLDER_PATHS)
    logger.info("Processo de tratamento de arquivos concluído.")
