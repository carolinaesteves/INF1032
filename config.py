# config.py

# BCB API Endpoints and Series IDs
BCB_ENDPOINT = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{}/dados?formato=csv"
IPEADATA_ENDPOINT = "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{}')"

# Series IDs for BCB
SERIES_IDS = {
    'selic_bcb': '432',
    'ipca_bcb': '433',
    'cambio_bcb': '432',
    'fx_buy_bcb': '10813',
    'fx_sell_bcb': '1',
    'gdp_bcb': '1207',
    'core_cpi_bcb': '11427',
    'commodities_bcb_composto_em_real': '27574',
    'commodities_bcb_agropecuaria_em_real': '27575',
    'commodities_bcb_metal_em_real': '27576',
    'commodities_bcb_energia_em_real': '27577',
    'unemployment_bcb': '24369'
}

# Series codes for Ipeadata
IPEADATA_SERIES_CODES = {
    'pib_series': 'PAN4_PIBPMG4',
}

YAHOO_FINANCE_TICKERS = {
    '^BVSP': {'start': '2000-01-01', 'end': '2023-12-31'},  # Ibovespa (Brasil)
    'BRL=X': {'start': '2000-01-01', 'end': '2023-12-31'},  # USD/BRL (Dólar)
    'CL=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Petróleo Bruto
    'GC=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Ouro
    'ZC=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Milho
    'KC=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Café
    'ZS=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Soja
    'ZW=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Trigo
    'HG=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Cobre
    'NG=F': {'start': '2000-01-01', 'end': '2023-12-31'},   # Gás Natural
    '^GSPC': {'start': '2000-01-01', 'end': '2023-12-31'},  # S&P 500
    '^TNX': {'start': '2000-01-01', 'end': '2023-12-31'},   # 10-Year Treasury Yield
    'BCOM': {'start': '2000-01-01', 'end': '2023-12-31'},   # Bloomberg Commodity Index
    'CPI': {'start': '2000-01-01', 'end': '2023-12-31'}     # Consumer Price Index (IPCA ou equivalente CPI dos EUA)
}



# CSV file base path for saving data
CSV_SAVE_PATHS = {
    'bcb': 'bcb_data_{}.csv',
    'ipeadata': 'ipeadata_{}.csv',
    'yahoo': 'yahoo_finance_{}.csv'
}

EXTRACTION_CSV_FOLDER_PATHS = "data/extraction/"
TREATMENT_CSV_FOLDER_PATHS = "data/treatment"
LOG_FOLDER_PATH = "log/"



COLUMNS_CONFIG = {
    'features': ['valor'],  # Colunas usadas como features
    'target': 'valor'       # Coluna alvo para previsão
}