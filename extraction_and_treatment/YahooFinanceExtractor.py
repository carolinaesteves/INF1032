import yfinance as yf
import pandas as pd

class YahooFinanceExtractor:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        """
        Fetch historical data for the specified ticker from Yahoo Finance.
        """
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        if data.empty:
            raise ValueError(f"No data found for ticker {self.ticker} in the given date range.")
        return data

    def process_data(self, data):
        """
        Process the raw Yahoo Finance data and return a cleaned DataFrame.
        """
        # Reset index to ensure 'Date' is a column
        data.reset_index(inplace=True)
        
        # Add a 'series' column with the ticker name
        data['series'] = self.ticker

        # Rename columns for consistency with other data sources
        data.rename(columns={
            'Date': 'data',
            'Close': 'valor'
        }, inplace=True)

        # We only need 'data', 'valor', and 'series'
        return data[['data', 'valor', 'series']]

    def get_processed_data(self):
        raw_data = self.fetch_data()
        return self.process_data(raw_data)
