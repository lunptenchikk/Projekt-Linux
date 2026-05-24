FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "hf_textgen.py", "--method", "temp", "--temp", "0.7", "--seed", "The ", "--len", "100"]