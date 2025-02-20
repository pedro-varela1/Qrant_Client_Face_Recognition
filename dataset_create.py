from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, UpdateStatus
from base_model import UltraLightORT, FaceNetORT
from PIL import Image, ImageOps
import numpy as np
import os
from configparser import ConfigParser
from typing import List, Union

# Importar variáveis de configuração para reconhecimento facial
config = ConfigParser()
config.read('config.ini')
MODEL_RECOGNITION_PATH = config.get('face_recognition', 'model_path')
MODEL_DETECTION_PATH = config.get('face_detection', 'model_path')
QDRANT_SERVER = config.get('face_recognition', 'qdrant_server_url')
QDRANT_COLLECTION = config.get('face_recognition', 'qdrant_collection_name')
FACEDIR = config.get('face_recognition', 'face_dir')
INPUT_SHAPE_RECOGNITION = tuple(map(int, config.get('face_recognition', 'input_shape').split(",")))
INPUT_SHAPE_DETECTION = tuple(map(int, config.get('face_detection', 'input_shape').split(",")))

# Modelos
model_detection = UltraLightORT(MODEL_DETECTION_PATH
                          # , "cuda"
                          )
model_facenet = FaceNetORT(MODEL_RECOGNITION_PATH
                            # , "cuda"
                            )

# Arquivo para armazenamento de embeddings
client = QdrantClient(url=QDRANT_SERVER)

def get_face(image_paths: List[str]) -> List[Image.Image]:
    batch = [
        np.expand_dims(
            np.array(
                Image.open(image_path).resize(INPUT_SHAPE_DETECTION, Image.Resampling.LANCZOS)
            ),
            axis=0,
        )
        for image_path in image_paths
    ]
    concatenated_batch = np.concatenate(batch, axis=0)
    boxes, batch_indices = model_detection(concatenated_batch)
    
    # Get the faces
    faces: List[Image.Image] = []
    for i, (box, batch_index) in enumerate(zip(boxes, batch_indices)):
        # Crop the face
        img = Image.open(image_paths[batch_index])
        img_width, img_height = img.size
        face = img.crop(
            (
                int(box[0] * img_width),
                int(box[1] * img_height),
                int(box[2] * img_width),
                int(box[3] * img_height),
            )
        )
        faces.append(face)
    return faces

def get_embedding(faces: List[Union[Image.Image, np.ndarray]]) -> List[List[float]]:
    batch = [
        np.expand_dims(
            np.array(
                face.resize(INPUT_SHAPE_RECOGNITION, Image.Resampling.LANCZOS) if isinstance(face, Image.Image) else Image.fromarray(face).resize(INPUT_SHAPE_RECOGNITION, Image.Resampling.LANCZOS)
            ),
            axis=0,
        )
        for face in faces
    ]
    concatenated_batch = np.concatenate(batch, axis=0)
    concatenated_batch = np.transpose(concatenated_batch, (0, 3, 1, 2))
    embeddings = model_facenet(concatenated_batch)[0]
    embeddings = embeddings / np.linalg.norm(embeddings)
    return embeddings.tolist()

def show_faces(faces: List[Image.Image], person_name: str = None):
    for face in faces:
        face.save(f"face_{person_name}.jpg")

def add_points(points: List[PointStruct]) -> UpdateStatus:
    print(f"Adicionando pontos à coleção {QDRANT_COLLECTION}")
    operation_info = client.upsert(
        collection_name=QDRANT_COLLECTION,
        wait=True,
        points=points,
    )
    return operation_info

def main():
    points = []
    # Carregando imagens
    for i, person in enumerate(os.listdir(FACEDIR)):
        person_name = person.split(".")[0]  # Person name is the filename without extension
        person_path = os.path.join(FACEDIR, person)

        # Face detection
        print()
        print(f"Detectando face para {person_name}")
        faces = get_face([person_path])

        # Rotacionando faces
        faces = [ImageOps.exif_transpose(face) for face in faces]

        # Mostrando faces
        # Se nome começar pr "LAB_" mostra as faces
        if person_name.startswith("LAB_"):
            show_faces(faces, person_name)

        # Face recognition
        print()
        print(f"Calculando embedding para {person_name}")
        try:
            embeddings = get_embedding(faces)
        except Exception as e:
            print(f"Erro ao calcular embedding para {person_name}: {str(e)}")
            continue
        
        # Adicionando ponto ao vetor de pontos
        points.append(PointStruct(
            id=i,
            vector=embeddings[0],
            payload={"person_name": person_name}
        ))

    try:
        operation_info = add_points(points)
    except Exception as e:
        print(f"Erro ao adicionar pontos à coleção: {str(e)}")
        return
    print()
    print(f"Status da operação: {operation_info.status}")

if __name__ == "__main__":
    main()
