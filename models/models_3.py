import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import pandas as pd
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from config import TREATMENT_CSV_FOLDER_PATHS
from utils.utilities import create_directory_if_not_exists
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score# Importar numpy
import matplotlib.pyplot as  plt
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

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

        # Iterar sobre os arquivos na pasta de arquivos tratados
        for filename in csv_files:
            file_path = os.path.join(user_path + TREATMENT_CSV_FOLDER_PATHS, filename)
            logger.info(f"Processando arquivo: {file_path}")
            
            # Carregar o CSV tratado
            df = pd.read_csv(file_path)

            # Verificar se as colunas necessárias estão presentes
            if 'data' in df.columns and 'valor' in df.columns:

                # Converter 'data' para datetime se ainda não foi feito
                df['data'] = pd.to_datetime(df['data'], errors='coerce')

                df['year'] = df['data'].dt.year
                df['month'] = df['data'].dt.month
                df['day'] = df['data'].dt.day
                df['weekday'] = df['data'].dt.weekday

                # Utilizar o valor numérico da data para o modelo preditivo
                df['data_numeric'] = df['data'].map(pd.Timestamp.toordinal)

                #df.to_excel(fr'C:\Users\Pichau\Projetos\Scripts\GitClones\INF1032\data\treatment\teste.xlsx')
                
                # Dividir os dados em conjunto de treino e teste

                X = df[['data_numeric', 'year', 'month', 'weekday']].dropna()
                y = df['valor'].loc[X.index]

                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


                # --------- Regressão Linear ----------
                
                # Instanciar e treinar o modelo de regressão linear
                model = LinearRegression()
                model.fit(X_train, y_train)
                
                # Fazer previsões
                y_pred = model.predict(X_test)
                
                # Calcular métricas de avaliação
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                print(f"\nArquivo: {filename}")
                print(f"Regr. Linear -> MSE: {mse:.5f} // RMSE: {rmse:.5f} // R²: {r2}")
                logger.info(f"Regr. Linear -> MSE: {mse:.5f} // RMSE: {rmse:.5f} // R²: {r2}")


                # --------- Regressão Random Forest ----------

                # Criar e treinar o modelo Random Forest para regressão
                rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
                rf_regressor.fit(X_train, y_train)

                # Fazer previsões
                y_pred_rf = rf_regressor.predict(X_test)

                # Calcular métricas de avaliação
                mse_rf = mean_squared_error(y_test, y_pred_rf)
                rmse_rf = np.sqrt(mse_rf)
                r2_rf = r2_score(y_test, y_pred_rf)

                print(f"Regr. Random Forest -> MSE: {mse_rf:.5f} // RMSE: {rmse_rf:.5f} // R²: {r2_rf}")
                logger.info(f"Regr. Random Forest -> MSE: {mse_rf:.5f} // RMSE: {rmse_rf:.5f} // R²: {r2_rf}")


                # --------- Regressão XGBoost ----------

                # Criar e treinar o modelo XGBoost para regressão
                xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, max_depth=3)
                xgb_regressor.fit(X_train, y_train)

                # Fazer previsões
                y_pred_xgb = xgb_regressor.predict(X_test)

                # Calcular métricas de avaliação
                mse_xgb = mean_squared_error(y_test, y_pred_xgb)
                rmse_xgb = np.sqrt(mse_xgb)
                r2_xgb = r2_score(y_test, y_pred_xgb)

                print(f"Regr. XGBoost -> MSE: {mse_xgb:.5f} // RMSE: {rmse_xgb:.5f} // R²: {r2_xgb}")
                logger.info(f"Regr. XGBoost -> MSE: {mse_xgb:.5f} // RMSE: {rmse_xgb:.5f} // R²: {r2_xgb}")


                # --------- Regressão Lasso ----------

                # Criar e treinar o modelo Lasso para regressão

                lasso_regressor = Lasso(alpha=0.01, random_state=42, max_iter=20000)
                lasso_regressor.fit(X_train, y_train)

                # Fazer previsões
                y_pred_lasso = lasso_regressor.predict(X_test)

                # Calcular métricas de avaliação
                mse_lasso = mean_squared_error(y_test, y_pred_lasso)
                rmse_lasso = np.sqrt(mse_lasso)
                r2_lasso = r2_score(y_test, y_pred_lasso)

                print(f"Lasso -> MSE: {mse_lasso:.5f} // RMSE: {rmse_lasso:.5f} // R²: {r2_lasso}")
                logger.info(f"Lasso -> MSE: {mse_lasso:.5f} // RMSE: {rmse_lasso:.5f} // R²: {r2_lasso}")


                # --------- Regressão Ridge ----------

                # Criar e treinar o modelo Ridge para regressão

                ridge_regressor = Ridge(alpha=100, random_state=42, max_iter=20000)
                ridge_regressor.fit(X_train, y_train)

                # Fazer previsões
                y_pred_ridge = ridge_regressor.predict(X_test)

                # Calcular métricas de avaliação
                mse_ridge = mean_squared_error(y_test, y_pred_ridge)
                rmse_ridge = np.sqrt(mse_ridge)
                r2_ridge = r2_score(y_test, y_pred_ridge)

                print(f"Ridge -> MSE: {mse_ridge:.5f} // RMSE: {rmse_ridge:.5f} // R²: {r2_ridge}")
                logger.info(f"Ridge -> MSE: {mse_ridge:.5f} // RMSE: {rmse_ridge:.5f} // R²: {r2_ridge}")


                # --------- Regressão Elastic Net ----------

                # Criar e treinar o modelo Elastic Net para regressão

                elastic_net_regressor = ElasticNet(alpha=0.01, l1_ratio=1, random_state=42, max_iter=20000)
                elastic_net_regressor.fit(X_train, y_train)

                # Fazer previsões
                y_pred_elastic_net = elastic_net_regressor.predict(X_test)

                # Calcular métricas de avaliação
                mse_elastic_net = mean_squared_error(y_test, y_pred_elastic_net)
                rmse_elastic_net = np.sqrt(mse_elastic_net)
                r2_elastic_net = r2_score(y_test, y_pred_elastic_net)

                print(f"Elastic Net -> MSE: {mse_elastic_net:.5f} // RMSE: {rmse_elastic_net:.5f} // R²: {r2_elastic_net}")
                logger.info(f"Elastic Net -> MSE: {mse_elastic_net:.5f} // RMSE: {rmse_elastic_net:.5f} // R²: {r2_elastic_net}")
                
            else:
                logger.warning(f"Colunas 'data' e 'valor' não encontradas no arquivo: {filename}")
        


        logger.info("Execução dos modelos preditivos concluída.")
    except Exception as e:
        logger.error(f"Erro durante a execução dos modelos: {str(e)}")

run_models()