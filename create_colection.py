from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from configparser import ConfigParser
import time

# Importar variáveis de configuração para reconhecimento facial
config = ConfigParser()
config.read('config.ini')
QDRANT_SERVER = config.get('face_recognition', 'qdrant_server_url')
QDRANT_COLLECTION = config.get('face_recognition', 'qdrant_collection_name')

client = QdrantClient(url=QDRANT_SERVER)

def create_collection(distance="cosine"):
    try:
        if client.collection_exists(QDRANT_COLLECTION):
            print(f"Coleção {QDRANT_COLLECTION} já existe")
            return
        print(f"Criando coleção {QDRANT_COLLECTION}")
    except Exception as e:
        print(f"Tente novamente, erro ao verificar coleções: {str(e)}")
        return
    if distance == "cosine":
        distance = Distance.COSINE
    else:
        distance = Distance.EUCLID

    try:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(
                size=128,
                distance=distance
            )
        )
    except Exception as e:
        print(f"Tentando novamente, erro ao criar coleção: {str(e)}")
        time.sleep(30)  # Espera 30 segundos
        try:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(
                    size=128,
                    distance=distance
                )
            )
        except Exception as e:
            print(f"Erro ao criar coleção: {str(e)}")
    return 

if __name__ == "__main__":
    create_collection(distance="euclid")