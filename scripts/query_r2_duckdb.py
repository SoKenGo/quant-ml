import os, duckdb
from dotenv import load_dotenv
from urllib.parse import urlparse

load_dotenv(".env")

endpoint_raw = os.getenv("R2_ENDPOINT")  # e.g. https://<ACCOUNT>.r2.cloudflarestorage.com
ak = os.getenv("R2_ACCESS_KEY_ID")
sk = os.getenv("R2_SECRET_ACCESS_KEY")
bucket = os.getenv("R2_BUCKET", "quant-ml")
assert all([endpoint_raw, ak, sk, bucket]), "Missing R2 env vars"

# Strip scheme for DuckDB s3_endpoint
parsed = urlparse(endpoint_raw)
endpoint_host = parsed.netloc or endpoint_raw.replace("https://", "").replace("http://", "")

con = duckdb.connect()
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region='auto';")
con.execute("SET s3_url_style='path';")
con.execute("SET s3_use_ssl=true;")
con.execute("SET s3_endpoint=?", [endpoint_host])      # host only, no scheme
con.execute("SET s3_access_key_id=?", [ak])
con.execute("SET s3_secret_access_key=?", [sk])

# Loosen the filter in case 2024 wasn't uploaded; you can narrow later
sql = f"""
SELECT date, adjClose, volume
FROM read_parquet('s3://{bucket}/eod/symbol=NVDA/year=*/part.parquet')
ORDER BY date
LIMIT 5
"""
print(con.execute(sql).df())
