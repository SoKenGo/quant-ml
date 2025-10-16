import os, requests, json
from dotenv import load_dotenv

load_dotenv(".env")
token = os.getenv("TIINGO_TOKEN")
assert token, "❌ Missing TIINGO_TOKEN in .env"

url = "https://api.tiingo.com/tiingo/daily/AAPL/prices"
r = requests.get(url, params={"token": token, "startDate": "2020-01-01"}, timeout=20)
print("HTTP", r.status_code)
data = r.json()
print(json.dumps(data[:2] if isinstance(data, list) else data, indent=2)[:400])
