
# eda.py
from venv import logger
import pandas as pd
import matplotlib.pyplot as plt
import logging

def run_eda():
    try:
        logger.info("Iniciando analise exploratoria de dados")
        #Carregar as bases de dados
        
        ibovespa = pd.read_csv('data/treatment/yahoo_finance_^BVSP.csv')
        cambio = pd.read_csv('data/treatment/bcb_data_cambio_bcb.csv')
        ipca = pd.read_csv('data/treatment/bcb_data_ipca_bcb.csv')
        pib = pd.read_csv('data/treatment/ipeadata_pib_series.csv')

        
        '''print(ibovespa.head())
        print(cambio.head())
        print(ipca.head())
        print(pib.head())
        
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
        print(pib.describe())'''
        
        
        # Converter a coluna 'data' para datetime
        ibovespa['data'] = pd.to_datetime(ibovespa['data'])
        cambio['data'] = pd.to_datetime(cambio['data'])
        ipca['data'] = pd.to_datetime(ipca['data'])
        pib['data'] = pd.to_datetime(pib['data'])
                
        # Merge dos datasets nas mesmas datas
        merged_df = pd.merge(ibovespa, cambio, on='data', how='inner', suffixes=('_ibovespa', '_cambio'))
        merged_df = pd.merge(merged_df, ipca, on='data', how='inner')
        merged_df = pd.merge(merged_df, pib, on='data', how='inner')
        
        # Renomear colunas para facilitar análise
        merged_df.columns = ['data', 'valor_ibovespa', 'series_ibovespa', 'valor_cambio', 'series_cambio', 
                             'valor_ipca', 'series_ipca', 'valor_pib', 'series_pib']

        # Plotar as séries temporais para uma visualização inicial
        # Plotar as séries temporais com dois eixos y
        fig, ax1 = plt.subplots(figsize=(14,8))

        # Eixo y1 (para o IBOVESPA)
        ax1.set_xlabel('Ano')
        ax1.set_ylabel('IBOVESPA', color='blue')
        ax1.plot(merged_df['data'], merged_df['valor_ibovespa'], label='IBOVESPA', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # Criar o segundo eixo y (para Câmbio, IPCA e PIB)
        ax2 = ax1.twinx()  
        ax2.set_ylabel('Câmbio, IPCA, PIB', color='orange')
        ax2.plot(merged_df['data'], merged_df['valor_cambio'], label='Câmbio', color='red')
        ax2.plot(merged_df['data'], merged_df['valor_ipca'], label='IPCA', color='green')
        ax2.plot(merged_df['data'], merged_df['valor_pib'], label='PIB', color='orange')
        ax2.tick_params(axis='y', labelcolor='orange')

        # Título e legendas
        fig.suptitle('Comparação das Séries Temporais: IBOVESPA, Câmbio, IPCA, PIB')
        fig.tight_layout()  # Ajusta para que os rótulos não fiquem sobrepostos
        fig.legend(loc='upper left')

        # Salvar o gráfico como arquivo .png
        plt.savefig('comparacao_series_temporais_com_dois_eixos.png')
        
        print("EDA realizada com sucesso.")


    except Exception as e:
        logger.error(f"Erro durante eda: {str(e)}")
