import os
import pandas as pd
import logging
from datetime import datetime
from config import TREATMENT_CSV_FOLDER_PATHS, COLUMNS_DATASET
from utils.utilities import create_directory_if_not_exists

def processar_e_unificar_dados(pasta_entrada, pasta_saida):
    """Processa e unifica os dados extraídos."""
    try:
        logging.info("Iniciando o processamento e a unificação dos dados...")
        data_frames = []

        for filename in os.listdir(pasta_entrada):
            if filename.endswith(".csv"):
                caminho_arquivo = os.path.join(pasta_entrada, filename)
                logging.info(f"Processando arquivo: {caminho_arquivo}")
                df = pd.read_csv(caminho_arquivo)
                df.columns = df.columns.str.strip()
                df[COLUMNS_DATASET['data']] = pd.to_datetime(df[COLUMNS_DATASET['data']], errors='coerce')
                df = df[[COLUMNS_DATASET['data'], COLUMNS_DATASET['valor'], COLUMNS_DATASET['series']]]
                data_frames.append(df)

        if not data_frames:
            raise ValueError("Nenhum arquivo válido foi processado.")

        dados_unificados = pd.concat(data_frames, axis=0)
        data_inicio = max([df[COLUMNS_DATASET['data']].min() for df in data_frames])
        data_fim = datetime.now().strftime('%Y-%m-%d')
        dados_unificados = dados_unificados[
            (dados_unificados[COLUMNS_DATASET['data']] >= data_inicio) &
            (dados_unificados[COLUMNS_DATASET['data']] <= data_fim)
        ]
        dados_unificados.sort_values(by=COLUMNS_DATASET['data'], inplace=True)
        create_directory_if_not_exists(pasta_saida, logging)
        caminho_saida = os.path.join(pasta_saida, "dados_unificados.csv")
        dados_unificados.to_csv(caminho_saida, index=False)
        logging.info(f"Base unificada salva em: {caminho_saida}")

        return dados_unificados
    except Exception as e:
        logging.error(f"Erro no processamento: {str(e)}")
        return None
