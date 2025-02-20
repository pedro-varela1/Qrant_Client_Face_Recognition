# Use Python 3.10.12 como imagem base
FROM python:3.10.12-slim

# Defina variáveis de ambiente para garantir que Python não crie arquivos .pyc
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Defina o diretório de trabalho no container
WORKDIR /app

# Copie apenas o arquivo requirements.txt primeiro
COPY requirements.txt .

# Instale as dependências do requirements.txt
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie o código do serviço
COPY face_recognition_service.py .
COPY config.ini .

# Exponha a porta do serviço Flask
EXPOSE 5000

# Comando para executar o serviço
CMD ["python", "face_recognition_service.py"]