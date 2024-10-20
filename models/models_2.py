import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from config import TREATMENT_CSV_FOLDER_PATHS
from utils.utilities import create_directory_if_not_exists
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error# Importar numpy
import matplotlib.pyplot as  plt

logger = logging.getLogger(__name__)

#user_path = 'C:/Users/Pichau/Projetos/Scripts/GitClones/INF1032/'
user_path = ''

def run_models():
    try:
        logger.info("Iniciando a execução dos modelos preditivos...")

        csv_files = [filename for filename in os.listdir(user_path + TREATMENT_CSV_FOLDER_PATHS) if filename.endswith('.csv')]

        if not csv_files:
            logger.warning('Nenhum arquivo CSV encontrado.')
            return

        first_file_path = os.path.join(user_path + TREATMENT_CSV_FOLDER_PATHS, csv_files[0])
        logger.info(f'Carregando o primeiro arquivo: {first_file_path}')
        merged_df = pd.read_csv(first_file_path)
        merged_df['data'] = pd.to_datetime(merged_df['data'], errors='coerce')
        sufixo = str(merged_df['series'].iloc[0])
        merged_df.rename(columns={
            'valor': f'valor_{sufixo}',
            'series': f'series_{sufixo}',
        }, inplace=True)
        #merged_df['data_numeric'] = merged_df['data'].map(pd.Timestamp.toordinal)


        # Iterar sobre os arquivos na pasta de arquivos tratados
        for filename in csv_files[1:]:
            file_path = os.path.join(user_path + TREATMENT_CSV_FOLDER_PATHS, filename)
            logger.info(f"Processando arquivo: {file_path}")
            
            # Carregar o CSV tratado
            df = pd.read_csv(file_path)

            # Verificar se as colunas necessárias estão presentes
            if 'data' in df.columns and 'valor' in df.columns:
                
                # Converter 'data' para datetime se ainda não foi feito
                df['data'] = pd.to_datetime(df['data'], errors='coerce')
                
                # Utilizar o valor numérico da data para o modelo preditivo
                #df['data_numeric'] = df['data'].map(pd.Timestamp.toordinal)

                sufixo = str(df['series'].iloc[0])
                df.rename(columns={
                    'valor': f'valor_{sufixo}',
                    'series': f'series_{sufixo}',
                }, inplace=True)

                merged_df = pd.merge(merged_df, df, on='data', how='inner')

            else:
                logger.warning(f"Colunas 'data' e 'valor' não encontradas no arquivo: {filename}")
        
        colunas_valor = [column for column in merged_df.columns if column.startswith('valor') and not column.endswith('^BVSP')]

        X = merged_df[colunas_valor]
        y = merged_df['valor_^BVSP']

        # Dividir os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Criar e treinar o modelo de regressão linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Fazer previsões
        y_pred = model.predict(X_test)

        # Avaliar o modelo
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)  # Calcular RMSE corretamente
        r2 = r2_score(y_test, y_pred)  # Certifique-se de que r2_score está importado

        print(f"MSE para merge dos arquivos: {mse}")
        print(f"RMSE para merge dos arquivos : {rmse}")
        print(f"R² para merge dos arquivos: {r2}")

        logger.info(f"MSE para merge dos arquivos: {mse}")
        logger.info(f"RMSE para merge dos arquivos : {rmse}")
        logger.info(f"R² para merge dos arquivos: {r2}")

        fig, ax = plt.subplots()

        ax.scatter(y_pred, y_test)
        ax.plot([30e3, 130e3], [30e3, 130e3], '--r')

        plt.show()
        
        logger.info("Execução dos modelos preditivos concluída.")
    except Exception as e:
        logger.error(f"Erro durante a execução dos modelos: {str(e)}")

run_models()