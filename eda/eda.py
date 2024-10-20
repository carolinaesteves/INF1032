
# eda.py
from venv import logger
import pandas as pd
import logging

def run_eda():
    try:
        logger.info("Iniciando analise exploratoria de dados")
        #Carregar as bases de dados
        
        ibovespa = pd.read_csv('data/treatment/yahoo_finance_^BVSP.csv')
        cambio = pd.read_csv('data/treatment/bcb_data_cambio_bcb.csv')
        ipca = pd.read_csv('data/treatment/bcb_data_ipca_bcb.csv')
        pib = pd.read_csv('data/treatment/ipeadata_pib_series.csv')
        print("EDA realizada com sucesso.")
        
        '''print(ibovespa.head())
        print(cambio.head())
        print(ipca.head())
        print(pib.head())'''
        
        print("IBOVESPA")
        print(ibovespa.head())
        print(ibovespa.describe())
        
        print("CAMBIO")
        print(cambio.head())
        print(cambio.describe())
        
        print("IPCA")
        print(ipca.head())
        print(ipca.describe())
        
        print("PIB")
        print(pib.head())
        print(pib.describe())
        
    except Exception as e:
        logger.error(f"Erro durante eda: {str(e)}")
