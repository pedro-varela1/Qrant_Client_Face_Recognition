# Use Python 3.10.12 como imagem base
FROM python:3.10.12-slim

# Defina variáveis de ambiente para garantir que Python não crie arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Defina o diretório de trabalho no container
WORKDIR /app

# Instale as dependências necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copie apenas o arquivo de requisitos primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instale as dependências específicas
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir qdrant-client==1.12.1 requests==2.32.3 flask==2.3.3 numpy==1.26.0

# Copie o código do serviço
COPY face_recognition_service.py .
COPY config.ini .

# Exponha a porta do serviço Flask
EXPOSE 5000

# Comando para executar o serviço
CMD ["python", "face_recognition_service.py"]
