import os
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from config import TREATMENT_CSV_FOLDER_PATHS
from utils.utilities import create_directory_if_not_exists

logger = logging.getLogger(__name__)

def run_models():
    try:
        logger.info("Iniciando a execução dos modelos preditivos...")

        # Iterar sobre os arquivos na pasta de arquivos tratados
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
        
        logger.info("Execução dos modelos preditivos concluída.")
    except Exception as e:
        logger.error(f"Erro durante a execução dos modelos: {str(e)}")

