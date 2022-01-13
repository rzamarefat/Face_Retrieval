import os

from MagFaceEmbeddingGenerator import MagFaceEmbeddingGenerator
from utils import Clustering, Reader
from config import config


class FaceRetrieval:
    def __init__(self):
        self.magface_embedding_generator = MagFaceEmbeddingGenerator()
        self.clustering = Clustering()
        self.reader = Reader()

        self._gallery_path = config["gallery_path"]
        self._path_to_save_embs = config["path_to_save_embs"]
        self._cpu_mode = config["cpu_mode"]
        self._magface_pretrained_path = config["magface_pretrained_path"]
        self._medoids_storage_path = config["medoids_storage_path"]

    def retreive_faces(self):
        data_path = self.reader.read_path(self._gallery_path)
        print(data_path)


if __name__ == "__main__":
    face_retriever = FaceRetrieval()

    face_retriever.retreive_faces()