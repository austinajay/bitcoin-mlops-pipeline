
import requests
import pandas as pd
import os

def fetch_daily_bitcoin_data():
    """Fetches the last 90 days of Bitcoin market data and engineers technical indicators."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "90", "interval": "daily"}
    
    try:
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        res_data = response.json()
        
        # Parse prices and volumes
        prices = res_data["prices"]
        volumes = res_data["total_volumes"]
        
        df_prices = pd.DataFrame(prices, columns=["timestamp", "price"])
        df_volumes = pd.DataFrame(volumes, columns=["timestamp", "volume"])
        
        df = pd.merge(df_prices, df_volumes, on="timestamp")
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
        
        # --- FEATURE ENGINEERING ---
        # 1. Daily Return
        df["price_return"] = df["price"].pct_change()
        # 2. Moving Averages
        df["sma_5"] = df["price"].rolling(window=5).mean()
        df["sma_10"] = df["price"].rolling(window=10).mean()
        # 3. Volatility
        df["volatility"] = df["price_return"].rolling(window=5).std()
        # 4. Volume Momentum
        df["volume_change"] = df["volume"].pct_change()
        
        # Create a simple target: 1 if price went up tomorrow, 0 if down
        df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
        
        # Drop rows with NaN from rolling calculations and get all historical days
        df = df.dropna()
        
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

if __name__ == "__main__":
    print("Running data ingestion...")
    df = fetch_daily_bitcoin_data()
    if df is not None:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/latest_ingested_data.csv", index=False)
        print(f"Data ingestion completed successfully. Ingested {len(df)} days of data.")