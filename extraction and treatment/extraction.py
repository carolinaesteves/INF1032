import pandas as pd
import requests
from io import StringIO

selic_bcb = '432'
fx_buy_bcb = "10813"
fx_sell_bcb = "1"
gdp_bcb = '1207'
core_cpi_bcb = '11427'


url = f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{core_cpi_bcb}/dados?formato=csv"


response = requests.get(url)
response_text = response.content.decode('utf-8')

# Step 3: Convert the string to a pandas DataFrame, specifying delimiter and decimal separator
data = StringIO(response_text)
df = pd.read_csv(data, delimiter='";"', engine='python')

# Optional: Clean up any leftover quotes (since the delimiter itself had quotes)
df.columns = df.columns.str.replace('"', '')  # Clean column names
df = df.replace('"', '', regex=True)  # Clean data entries
df['data'] = pd.to_datetime(df['data'], format="%d/%m/%Y")


# Step 4: Print the DataFrame or perform further operations
print(df)
