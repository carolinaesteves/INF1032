import os
import pandas as pd
import logging
from config import EXTRACTION_CSV_FOLDER_PATHS, TREATMENT_CSV_FOLDER_PATHS

logger = logging.getLogger(__name__)

def treat_ipeadata_file(file_path, output_folder):
    try:
        logger.info(f"Processando arquivo Ipeadata: {file_path}")
        
        # Load CSV file
        df = pd.read_csv(file_path)
        logger.info(f"Arquivo CSV carregado com sucesso: {file_path}")
        
        # Check for required columns and rename them
        if 'SERCODIGO' in df.columns and 'Date' in df.columns and 'Value' in df.columns:
            df = df[['Date', 'SERCODIGO', 'Value']]
            df.rename(columns={'Date': 'data', 'SERCODIGO': 'series', 'Value': 'valor'}, inplace=True)
        else:
            raise ValueError("O arquivo CSV não contém as colunas esperadas: 'SERCODIGO', 'Date', 'Value'.")
        
        # Convert 'data' column to datetime and normalize time to 00:00:00
        df['data'] = pd.to_datetime(df['data'], errors='coerce').dt.normalize()
        
        # Get the unique value in 'series'
        series_value = df['series'].dropna().unique()
        if len(series_value) != 1:
            raise ValueError("Valores múltiplos inesperados na coluna 'series'.")
        series_value = series_value[0]

        # Set 'data' as index
        df.set_index('data', inplace=True)
        
        # Reindex to fill gaps with daily dates
        logger.info("Reindexando as datas para preencher lacunas com datas diárias.")
        df_reindexed = df.resample('D').asfreq()
        
        # Preserve original 'valor' values and interpolate missing ones
        df_reindexed['valor'] = df_reindexed['valor'].combine_first(df['valor'])
        df_reindexed['valor'] = df_reindexed['valor'].interpolate(method='linear', limit_direction='both')
        
        # Set the same 'series' value for all records
        df_reindexed['series'] = series_value
        
        # Reset index and convert 'data' back to string format
        df_reindexed.reset_index(inplace=True)
        df_reindexed['data'] = df_reindexed['data'].dt.strftime('%Y-%m-%d')

        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Save the treated file
        treated_file_path = os.path.join(output_folder, os.path.basename(file_path))
        df_reindexed.to_csv(treated_file_path, index=False)
        
        logger.info(f"Arquivo tratado salvo com sucesso: {treated_file_path}")
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo Ipeadata: {file_path} - {str(e)}")


def treat_bcbdata_file(file_path, output_folder):
    try:
        logger.info(f"Processando arquivo BCB: {file_path}")
        
        # Load CSV file
        df = pd.read_csv(file_path)
        
        # Check for required columns
        expected_columns = ['data', 'valor', 'series']
        missing_columns = [col for col in expected_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Faltando as seguintes colunas: {missing_columns}")

        # Convert 'data' column to datetime
        df['data'] = pd.to_datetime(df['data'], errors='coerce')
        if df['data'].isnull().any():
            logger.warning("Valores nulos encontrados na coluna 'data'.")
        
        # Set 'data' as index
        df.set_index('data', inplace=True)
        
        # Reindex to fill gaps with daily dates
        df = df.resample('D').asfreq()
        
        # Interpolate missing values in 'valor'
        df['valor'] = df['valor'].interpolate(method='linear')
        
        # Reset index and convert 'data' back to string format
        df.reset_index(inplace=True)
        df['data'] = df['data'].dt.strftime('%Y-%m-%d')

        # Fill missing 'series' values
        valid_series_value = df['series'].dropna().unique()
        if len(valid_series_value) == 1:
            df['series'] = df['series'].fillna(valid_series_value[0])
        else:
            df['series'] = df['series'].ffill().bfill()

        df['series'] = df['series'].astype(str)

        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # Save the treated file
        treated_file_path = os.path.join(output_folder, os.path.basename(file_path))
        df.to_csv(treated_file_path, index=False)

        logger.info(f"Arquivo BCB tratado salvo com sucesso: {treated_file_path}")
    except Exception as e:
        logger.error(f"Erro ao processar o arquivo BCB: {file_path} - {str(e)}")


def treatment():
    logger.info("Iniciando o processo de tratamento de arquivos...")
    for filename in os.listdir(EXTRACTION_CSV_FOLDER_PATHS):
        if filename.endswith(".csv"):
            file_path = os.path.join(EXTRACTION_CSV_FOLDER_PATHS, filename)
            logger.info(f"Processando arquivo: {file_path}")
            if "ipeadata" in filename.lower():
                treat_ipeadata_file(file_path, TREATMENT_CSV_FOLDER_PATHS)
            else:
                treat_bcbdata_file(file_path, TREATMENT_CSV_FOLDER_PATHS)
    logger.info("Processo de tratamento de arquivos concluído.")
