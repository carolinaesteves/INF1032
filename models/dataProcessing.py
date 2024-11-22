import os
import pandas as pd
import logging
from datetime import datetime, timedelta
from config import TREATMENT_CSV_FOLDER_PATHS, COLUMNS_DATASET
from utils.utilities import create_directory_if_not_exists


def processar_e_unificar_dados(pasta_entrada, pasta_saida):
    """
    Lê, processa e unifica arquivos CSV de uma pasta, aplicando filtros de data e salvando os resultados.

    Args:
        pasta_entrada (str): Caminho da pasta onde os arquivos CSV estão localizados.
        pasta_saida (str): Caminho da pasta onde o arquivo unificado será salvo.

    Returns:
        pd.DataFrame: DataFrame unificado contendo os dados processados.
    """
    try:
        logging.info("Iniciando o processamento e a unificação dos dados...")

        # Lista para armazenar os DataFrames de cada arquivo processado
        data_frames = []

        # Itera sobre todos os arquivos da pasta
        for filename in os.listdir(pasta_entrada):
            # Verifica se o arquivo tem a extensão .csv
            if filename.endswith(".csv"):
                caminho_arquivo = os.path.join(pasta_entrada, filename)
                logging.info(f"Processando arquivo: {caminho_arquivo}")
                
                # Lê o arquivo CSV
                df = pd.read_csv(caminho_arquivo)
                
                # Remove espaços desnecessários dos nomes das colunas
                df.columns = df.columns.str.strip()
                
                # Converte a coluna de datas para o tipo datetime
                df[COLUMNS_DATASET['data']] = pd.to_datetime(df[COLUMNS_DATASET['data']], errors='coerce')
                
                # Seleciona apenas as colunas de interesse
                df = df[[COLUMNS_DATASET['data'], COLUMNS_DATASET['valor'], COLUMNS_DATASET['series']]]
                
                # Adiciona o DataFrame à lista
                data_frames.append(df)

        # Verifica se nenhum arquivo foi processado
        if not data_frames:
            raise ValueError("Nenhum arquivo válido foi processado.")

        # Combina todos os DataFrames em um único DataFrame
        dados_unificados = pd.concat(data_frames, axis=0)

        # Define o intervalo de datas com base nos arquivos
        data_inicio = max([df[COLUMNS_DATASET['data']].min() for df in data_frames])  # Data inicial mais recente
        data_fim = datetime.now().strftime('%Y-%m-%d')  # Data final como hoje

        # Filtra os dados para manter apenas os que estão dentro do intervalo definido
        dados_unificados = dados_unificados[
            (dados_unificados[COLUMNS_DATASET['data']] >= data_inicio) & 
            (dados_unificados[COLUMNS_DATASET['data']] <= data_fim)
        ]

        # Ordena os dados pela coluna de data
        dados_unificados.sort_values(by=COLUMNS_DATASET['data'], inplace=True)

        # Garante que a pasta de saída exista
        create_directory_if_not_exists(pasta_saida, logging)

        # Salva os dados unificados em um arquivo CSV
        caminho_saida = os.path.join(pasta_saida, "dados_unificados.csv")
        dados_unificados.to_csv(caminho_saida, index=False)
        logging.info(f"Base unificada salva em: {caminho_saida}")

        # Retorna o DataFrame unificado
        return dados_unificados
    except Exception as e:
        logging.error(f"Erro no processamento: {str(e)}")
        return None


def gerar_dados_futuros(dados, dias_previsao=60, janela=30):
    """
    Gera previsões de dados futuros usando a média móvel.

    Args:
        dados (pd.DataFrame): DataFrame contendo os dados históricos.
        dias_previsao (int): Número de dias futuros a serem gerados.
        janela (int): Tamanho da janela da média móvel.

    Returns:
        pd.DataFrame: DataFrame contendo os dados futuros gerados.
    """
    logging.info("Gerando dados futuros com base na média móvel...")
    
    # Lista para armazenar os dados futuros
    series_futuras = []
    
    # Obtém a última data dos dados históricos
    ultima_data = dados[COLUMNS_DATASET['data']].max()

    # Itera sobre cada série única nos dados
    for serie in dados['series'].unique():
        # Filtra os dados para a série atual
        df_serie = dados[dados['series'] == serie]
        
        # Define a coluna de data como índice e ordena os dados
        df_serie = df_serie.set_index(COLUMNS_DATASET['data']).sort_index()
        
        # Calcula a média móvel com a janela definida
        df_serie['media_movel'] = df_serie[COLUMNS_DATASET['valor']].rolling(window=janela, min_periods=1).mean()

        # Gera os dados futuros para os próximos dias
        for i in range(1, dias_previsao + 1):
            nova_data = ultima_data + timedelta(days=i)  # Adiciona dias à última data
            media_prevista = df_serie['media_movel'].iloc[-1]  # Usa a última média móvel calculada
            series_futuras.append({
                COLUMNS_DATASET['data']: nova_data, 
                COLUMNS_DATASET['valor']: media_prevista, 
                COLUMNS_DATASET['series']: serie
            })

    # Retorna os dados futuros como um DataFrame
    df_futuro = pd.DataFrame(series_futuras)
    return df_futuro


def unificar_com_dados_futuros(dados_unificados, dados_futuros, pasta_saida):
    """
    Combina os dados reais com os dados futuros gerados.

    Args:
        dados_unificados (pd.DataFrame): DataFrame contendo os dados reais.
        dados_futuros (pd.DataFrame): DataFrame contendo os dados futuros.
        pasta_saida (str): Caminho da pasta onde o arquivo combinado será salvo.

    Returns:
        pd.DataFrame: DataFrame contendo os dados reais e futuros combinados.
    """
    logging.info("Unificando dados reais com futuros...")

    # Combina os dados unificados (reais) com os dados futuros
    dados_completos = pd.concat([dados_unificados, dados_futuros], axis=0)
    
    # Ordena os dados pela coluna de data
    dados_completos.sort_values(by=COLUMNS_DATASET['data'], inplace=True)

    # Salva o DataFrame combinado em um arquivo CSV
    caminho_saida = os.path.join(pasta_saida, "dados_completos.csv")
    dados_completos.to_csv(caminho_saida, index=False)
    logging.info(f"Dados unificados com previsões futuras salvos em: {caminho_saida}")

    # Retorna o DataFrame combinado
    return dados_completos
