
import requests
import pandas as pd
import os

def fetch_daily_bitcoin_data():
    """Fetches the last 365 days of Bitcoin market data and engineers technical indicators."""
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": "365", "interval": "daily"}
    
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
        
        # --- FEATURE ENGINEERING (STATIONARY INDICATORS) ---
        # 1. Daily Returns
        df["price_return"] = df["price"].pct_change()
        df["price_return_2d"] = df["price"].pct_change(2)
        df["price_return_3d"] = df["price"].pct_change(3)
        
        # 2. Moving Averages
        df["sma_5"] = df["price"].rolling(window=5).mean()
        df["sma_10"] = df["price"].rolling(window=10).mean()
        df["sma_20"] = df["price"].rolling(window=20).mean()
        
        # 3. Price-to-MA Ratios
        df["price_sma_5_ratio"] = df["price"] / df["sma_5"] - 1.0
        df["price_sma_10_ratio"] = df["price"] / df["sma_10"] - 1.0
        df["price_sma_20_ratio"] = df["price"] / df["sma_20"] - 1.0
        
        # 4. Volatility
        df["volatility_5"] = df["price_return"].rolling(window=5).std()
        df["volatility_10"] = df["price_return"].rolling(window=10).std()
        
        # 5. Volume change and Volume-to-MA Ratio
        df["volume_change"] = df["volume"].pct_change()
        df["volume_sma_5"] = df["volume"].rolling(window=5).mean()
        df["volume_sma_5_ratio"] = df["volume"] / df["volume_sma_5"] - 1.0
        
        # 6. RSI (centered around 0)
        def compute_rsi(series, period=14):
            delta = series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        df["rsi_14"] = compute_rsi(df["price"], 14) / 100.0 - 0.5
        
        # Create target: 1 if price goes up tomorrow, 0 if down
        df["target"] = (df["price"].shift(-1) > df["price"]).astype(int)
        
        # Drop rows with NaN from rolling calculations
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