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
    'commodities_bcb_a': '27574',
    'commodities_bcb_b': '27575',
    'commodities_bcb_c': '27576',
    'commodities_bcb_d': '27577',
    'unemployment_bcb': '24369'
}

# Series codes for Ipeadata
IPEADATA_SERIES_CODES = {
    'pib_series': 'PAN4_PIBPMG4',
}

# CSV file base path for saving data
CSV_SAVE_PATHS = {
    'bcb': "bcb_data_{}.csv",  # Example for BCB data
    'ipeadata': "ipeadata_{}.csv",  # Example for Ipeadata data
}

EXTRACTION_CSV_FOLDER_PATHS = "data/extraction/"
