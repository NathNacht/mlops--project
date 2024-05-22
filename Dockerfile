FROM python:3.8

# Install MLflow and its dependencies
RUN pip install mlflow boto3 sqlalchemy psycopg2-binary

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . /opt/

WORKDIR /opt

EXPOSE 8000 5000

# Set entrypoint for MLflow
CMD ["bash", "-c", "mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000"]