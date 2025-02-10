import requests
import pandas as pd

from airflow.decorators import dag, task
from datetime import datetime, timedelta


# Global variables
TICKER = "XXBTZUSD"
START_DATE = datetime(2025, 1, 1)
RETRY_DELAY = timedelta(minutes=5)

# Define the DAG
@dag(
    # schedule="@daily",
    start_date=START_DATE,
    catchup=False,
    tags=["BTC"],
)
def api_to_csv_dag():
    
    # Extract: Fetch data from Kraken API
    @task()
    def extract_market_data():
        # Daily interval
        interval = str(60*24)

        # Construct the API URL for fetching OHLC data
        url = f"https://api.kraken.com/0/public/OHLC?pair={TICKER}&interval={interval}"

        # Set up request headers
        headers = {'Accept': 'application/json'}

        try:
            # Make the request to the Kraken API
            response = requests.request(
                "GET", url, headers=headers,
                data={}, timeout=10
            )
        except requests.exceptions.Timeout as e:
            # Print timeout exception message
            print(e)

        # subset of the json
        json_data = response.json()['result'][TICKER]
        
        return json_data
    
    # Transform: Parse the JSON response into a DataFrame
    @task()
    def transform_to_df(json):
    
        # Parse the JSON response into a DataFrame
        df = pd.DataFrame(
            json,
            columns=['timestamp', 'open', 'high', 'low',
                        'close', 'vwap', 'volume', 'count'],
            dtype=float
        )

        # Convert timestamps to datetime and set as index
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        return df
    
    # Load: Save DataFrame to CSV
    @task()
    def load_to_csv(df):
        
        # save to csv
        today = datetime.now().strftime("%Y-%m-%d")
        file_path = f"btc_data_{today}.csv"
        
        df.to_csv(
            path_or_buf=file_path,
            index=False
        )
        
        print(f"csv saved on {today} to {file_path}")
    
    
    # Start main flow
    json_data = extract_market_data()
    df = transform_to_df(json_data)
    load_to_csv(df)
    
# Run the DAG
api_to_csv_dag()
