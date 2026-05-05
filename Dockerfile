FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

COPY raw_data/ ./raw_data

CMD ["python", "src/main.py"]