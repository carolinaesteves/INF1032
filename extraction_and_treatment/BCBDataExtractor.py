import requests
import pandas as pd
from io import StringIO

class BCBDataExtractor:
    def __init__(self, series_id, series_name, base_url):
        self.series_id = series_id
        self.series_name = series_name
        self.base_url = base_url

    def fetch_data(self):
        url = self.base_url.format(self.series_id)
        response = requests.get(url)
        if response.status_code == 200:
            return response.content.decode('utf-8')
        else:
            raise ValueError(f"Error fetching data: {response.status_code}")

    def process_data(self, raw_data):
        # Step 1: Convert raw string data to pandas DataFrame
        data = StringIO(raw_data)
        df = pd.read_csv(data, delimiter=';', engine='python')

        # Step 2: Clean up the 'valor' column, and convert 'data' to datetime
        df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y", dayfirst=True)  # Adjust for day/month/year format
        df['valor'] = df['valor'].str.replace(",", ".").astype(float)  # Convert 'valor' to numeric

        # Step 3: Add a new column indicating the series name
        df['series'] = self.series_name

        return df

    def get_processed_data(self):
        raw_data = self.fetch_data()
        return self.process_data(raw_data)
