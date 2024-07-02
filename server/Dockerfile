
FROM python:3.11-slim

WORKDIR .

COPY requirements.txt ./app
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . ./app

EXPOSE 80


CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
