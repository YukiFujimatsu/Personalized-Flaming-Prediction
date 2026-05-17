# 軽量なPython環境をベースにする
FROM python:3.9-slim

# 作業ディレクトリの設定
WORKDIR /app

# 必要なパッケージ（ライブラリ）のインストール
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# アプリのコードとデータをコンテナにコピー
COPY main.py .
COPY data/ ./data/

# FastAPIを起動するコマンド
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]