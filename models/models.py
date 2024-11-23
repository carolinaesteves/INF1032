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
import plotly.graph_objects as go
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)

def plot_plt(dataframe, column, graph_title, line_label):
    plt.plot(dataframe.index,dataframe[column])
    plt.show()

def plot(dataframe, graph_title):
    gridcolor = 'rgba(225,226,220,1)'

    xaxis = dict(title="<b></b>",
             titlefont=dict(family='Arial', size=18, color='rgb(00,00,00)'),  # size was 16
             tickmode='auto',
             tickformat="%b-%Y")

    yaxis = dict(title="<b></b>",
                titlefont=dict(family='Arial', size=18, color='rgb(00,00,00)'),  # size was 16
                tickformat=".2%",
                overlaying='y2',
                showgrid=True,
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='Black',
                gridcolor=gridcolor)
    layout = go.Layout(title=dict(text='',
                              x=0.5,
                              font=dict(family='arial', size=20),
                              xanchor='center',
                              yanchor='top'),
                   hovermode="x",
                   font=dict(family='arial', size=13.5),  # size was 12
                   legend=dict(orientation="h", font=dict(size=14.5)),  # size was 12
                   showlegend=True,
                   xaxis=xaxis,
                   yaxis=yaxis,
                   plot_bgcolor='rgba(0,0,0,0)',
                   )

    fig = go.Figure()
    fig.layout = layout
    fig.layout.yaxis.tickformat = ".2%"
    fig.layout.xaxis.tickformat = ".2%"
    model_coef = [ 0.02529287, -0.00374899, -0.00225704]
    model_intr = 0.0050265308299753874
    X = dataframe['ibovespa'] #, 'fx_buy_bcb', 'ibovespa'
    Y = dataframe['ipca_bcb']
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="markers",
        ))
    Y = model_coef[2]*X + model_intr
    fig.add_trace(
        go.Scatter(
            x=X,
            y=Y,
            mode="lines",
            opacity = 0.3
        ))
    
    fig.layout.title.text = graph_title 
    return fig


def resample_df(df):
    # Passo 1: Resample para frequência mensal (último valor de cada mês)
    df = df.set_index('data')
    df_mensal = df.resample('M').last()

    # Passo 2: Calcular a variação percentual em relação ao mês anterior
    df_mensal['variacao_percentual'] = df_mensal['valor'].pct_change()

    # Passo 3: Ajustar a data para o primeiro dia de cada mês
    df_mensal.index = df_mensal.index.to_period('M').to_timestamp()
    return df_mensal

def run_models():
    try:
        logger.info("Iniciando a execução dos modelos preditivos...")

        # Carregar as bases de dados
        ibovespa = pd.read_csv('data/treatment/yahoo_finance_^BVSP.csv')
        cambio = pd.read_csv('data/treatment/bcb_data_fx_buy_bcb.csv')
        selic = pd.read_csv('data/treatment/bcb_data_selic_bcb.csv')
        ipca = pd.read_csv('data/treatment/bcb_data_ipca_bcb.csv')
        pib = pd.read_csv('data/treatment/ipeadata_pib_series.csv')

        # Converter a coluna 'data' para datetime
        ibovespa['data'] = pd.to_datetime(ibovespa['data'])
        cambio['data'] = pd.to_datetime(cambio['data'])
        selic['data'] = pd.to_datetime(selic['data'])
        ipca['data'] = pd.to_datetime(ipca['data'])
        pib['data'] = pd.to_datetime(pib['data'])

        ipca['valor'] /= 100


        ibovespa = resample_df(ibovespa)
        cambio = resample_df(cambio)
        selic = resample_df(selic)
        ipca = ipca.set_index('data')
        # pib = resample_df(pib)
        

        ibovespa = ibovespa.rename(columns={"variacao_percentual":"ibovespa"})
        cambio = cambio.rename(columns={"variacao_percentual":"fx_buy_bcb"})
        selic = selic.rename(columns={"variacao_percentual":"selic_bcb"})
        ipca = ipca.rename(columns={"valor":"ipca_bcb"})
        # pib = pib.rename(columns={"variacao_percentual":"PAN4_PIBPMG4"})

        # fig = plot(ibovespa,"ibovespa","Variação mensal do Ibovespa", "Ibovespa")
        # fig2 = plot(cambio,"fx_buy_bcb","Variação mensal do cambio USD/BRL", "USD/BRL")
        # fig3 = plot(selic,"selic_bcb","Variação mensal da taxa Selic", "Selic")
        # fig4 = plot(ipca,"ipca_bcb","Variação mensal do IPCA", "IPCA")

        # plot_plt(ibovespa,"ibovespa","Variação mensal do Ibovespa", "ibovespa")
        # plot_plt(cambio,"fx_buy_bcb","Variação mensal do cambio USD/BRL", "USD/BRL")
        # plot_plt(selic,"selic_bcb","Variação mensal da taxa Selic", "Selic")
        # plot_plt(ibovespa,"ipca_bcb","Variação mensal do IPCA", "IPCA")


        ibovespa.drop(['series','valor'],axis=1,inplace=True)
        cambio.drop(['series','valor'],axis=1,inplace=True)
        selic.drop(['series','valor'],axis=1,inplace=True)
        ipca.drop(['series'],axis=1,inplace=True)
        pib.drop(['series','valor'],axis=1,inplace=True)

        # Merge dos datasets nas mesmas datas
        merged_df = pd.merge(ibovespa, cambio, on='data', how='left')
        merged_df = pd.merge(merged_df, ipca, on='data', how='left')
        # merged_df = pd.merge(merged_df, pib, on='data', how='left')
        merged_df = pd.merge(merged_df, selic, on='data', how='left')
        merged_df = merged_df.dropna()
        # fig = plot(merged_df,"Variação mensal")
        # fig.show()

# Preparar os dados para regressão
        
        X = merged_df[['selic_bcb', 'fx_buy_bcb', 'ibovespa']]  # Variáveis independentes
        y = merged_df['ipca_bcb']  # Variável dependente

        # Dividir os dados em conjuntos de treinamento e teste
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Criar e treinar o modelo de regressão linear
        model = LinearRegression()
        model.fit(X_train, y_train)

        model_coef = [ 0.02529287, -0.00374899, -0.00225704]
        model_intr = 0.0050265308299753874
        fig = plot(merged_df,"Regressão Linear Ibovespa vs IPCA")
        fig.show()

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

