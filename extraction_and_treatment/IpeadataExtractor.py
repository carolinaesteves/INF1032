import requests
import pandas as pd

class IpeadataExtractor:
    def __init__(self, series_code, base_url):
        self.series_code = series_code
        self.base_url = base_url

    def fetch_data(self):
        url = self.base_url.format(self.series_code)
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Error fetching data: {response.status_code}")

    def process_data(self, raw_data):
        # Extract the 'value' part from the JSON response
        data = raw_data['value']

        # Convert to a pandas DataFrame
        df = pd.DataFrame(data)

        # Convert 'VALDATA' to datetime and specify utc=True to silence the warning
        df['VALDATA'] = pd.to_datetime(df['VALDATA'], utc=True)

        # Clean up column names if necessary
        df = df.rename(columns={'VALDATA': 'Date', 'VALVALOR': 'Value'})

        return df

    def get_processed_data(self):
        raw_data = self.fetch_data()
        return self.process_data(raw_data)
