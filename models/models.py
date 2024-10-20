import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from config import TREATMENT_CSV_FOLDER_PATHS
from utils.utilities import create_directory_if_not_exists
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score# Importar numpy

logger = logging.getLogger(__name__)

def run_models():
    try:
        logger.info("Iniciando a execução dos modelos preditivos...")

        '''# Iterar sobre os arquivos na pasta de arquivos tratados
        for filename in os.listdir(TREATMENT_CSV_FOLDER_PATHS):
            if filename.endswith(".csv"):
                file_path = os.path.join(TREATMENT_CSV_FOLDER_PATHS, filename)
                logger.info(f"Processando arquivo: {file_path}")
                
                # Carregar o CSV tratado
                df = pd.read_csv(file_path)
                
                # Verificar se as colunas necessárias estão presentes
                if 'data' in df.columns and 'valor' in df.columns:
                    

                    # COLOCAR OS MODELOS QUE SERÃO USADOS



                    # Converter 'data' para datetime se ainda não foi feito
                    df['data'] = pd.to_datetime(df['data'], errors='coerce')
                    
                    # Utilizar o valor numérico da data para o modelo preditivo
                    df['data_numeric'] = df['data'].map(pd.Timestamp.toordinal)

                    print(df.head())
                    
                    # Dividir os dados em conjunto de treino e teste
                    X = df[['data_numeric']]
                    y = df['valor']
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    
                    # Instanciar e treinar o modelo de regressão linear
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    
                    # Fazer previsões
                    y_pred = model.predict(X_test)
                    
                    # Calcular métricas de avaliação
                    mse = mean_squared_error(y_test, y_pred)
                    logger.info(f"MSE para o arquivo {filename}: {mse}")
                else:
                    logger.warning(f"Colunas 'data' e 'valor' não encontradas no arquivo: {filename}")
        '''
        
        # Carregar as bases de dados
        ibovespa = pd.read_csv('data/treatment/yahoo_finance_^BVSP.csv')
        cambio = pd.read_csv('data/treatment/bcb_data_cambio_bcb.csv')
        ipca = pd.read_csv('data/treatment/bcb_data_ipca_bcb.csv')
        pib = pd.read_csv('data/treatment/ipeadata_pib_series.csv')

        # Converter a coluna 'data' para datetime
        ibovespa['data'] = pd.to_datetime(ibovespa['data'])
        cambio['data'] = pd.to_datetime(cambio['data'])
        ipca['data'] = pd.to_datetime(ipca['data'])
        pib['data'] = pd.to_datetime(pib['data'])
        
        # Merge dos datasets nas mesmas datas
        merged_df = pd.merge(ibovespa, cambio, on='data', how='inner', suffixes=('_ibovespa', '_cambio'))
        merged_df = pd.merge(merged_df, ipca, on='data', how='inner')
        merged_df = pd.merge(merged_df, pib, on='data', how='inner')
        
        merged_df.columns = ['data', 'valor_ibovespa', 'series_ibovespa', 'valor_cambio', 'series_cambio', 'valor_ipca', 'series_ipca', 'valor_pib', 'series_pib']

# Preparar os dados para regressão
        X = merged_df[['valor_cambio', 'valor_ipca', 'valor_pib']]  # Variáveis independentes
        y = merged_df['valor_ibovespa']  # Variável dependente

        # Dividir os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Criar e treinar o modelo de regressão linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Avaliar o modelo
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Calcular RMSE corretamente
        r2 = r2_score(y_test, y_pred)  # Certifique-se de que r2_score está importado

        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")
        r2 = r2_score(y_test, y_pred)
        
        print(f"RMSE: {rmse}")
        print(f"R²: {r2}")

        logger.info("Execução dos modelos preditivos concluída.")
    except Exception as e:
        logger.error(f"Erro durante a execução dos modelos: {str(e)}")

