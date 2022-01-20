from decimal import Clamped
from dis import dis
import os
from turtle import distance

from MagFaceEmbeddingGenerator import MagFaceEmbeddingGenerator
from utils import Clustering, Reader, Preprocess
from config import config
import numpy as np
import cv2


class FaceRetrieval:
    def __init__(self, K=3):
        self.K = K
        self.magface_embedding_generator = MagFaceEmbeddingGenerator()
        self.clustering = Clustering(self.K)
        self.reader = Reader()
        self.preprocessor = Preprocess()
        

        
        self._gallery_path = config["gallery_path"]
        self._path_to_save_embs = config["path_to_save_embs"]
        self._cpu_mode = config["cpu_mode"]
        self._magface_pretrained_path = config["magface_pretrained_path"]
        self._medoids_storage_path = config["medoids_storage_path"]

    def do_preprocessing(self):
        img_data_path = self.reader.read_path(self._gallery_path)
        embs_data_path = self.reader.read_path(config["path_to_save_embs"])

        
        for class_name, img_path in img_data_path:
            img = cv2.imread(img_path)
            img = self.preprocessor.resize_img(img)
            
            abs_path_to_save = os.path.join(self._path_to_save_embs, class_name, f'{img_path.split("/")[-1].split(".")[0]}.npy')
            self.magface_embedding_generator.generate_embeddings(img, abs_path_to_save)

        obtained_classes = []
        data_to_feed_clustering = {}

        for class_name, data_path in embs_data_path:
            
            if not(class_name in obtained_classes):
                obtained_classes.append(class_name)
                data_to_feed_clustering[class_name] = []
            
            print(class_name)
            emb = self.reader.read_npy(data_path)
            data_to_feed_clustering[class_name].append(emb)
        

        for class_name, embs in data_to_feed_clustering.items():
            print("len(embs)", len(embs))
            medoids = self.clustering.get_medoids(np.array(embs))
            
            if not(os.path.isdir(os.path.join(config["medoids_storage_path"], class_name))): 
                os.mkdir(os.path.join(config["medoids_storage_path"], class_name))
            
            for index, m in enumerate(medoids):
                abs_path_to_save = os.path.join(config["medoids_storage_path"], class_name, f"{index}.npy")
                print(abs_path_to_save)
                self.clustering.save_medoid(m, abs_path_to_save)

        # Just getting the tARGWT

    def retrieve_face(self, path_to_face):
        obtained_classes = []
        medoids_path_data = self.reader.read_path(config["medoids_storage_path"])

        img = cv2.imread(path_to_face)
        img = self.preprocessor.resize_img(img)
            
        target_emb = self.magface_embedding_generator.generate_embeddings(img, abs_path_to_save_emb=None)


        
        obtained_classes = []
        history_of_distances = {}
        for class_name, data_path in medoids_path_data:
            if not(class_name in obtained_classes):
                obtained_classes.append(class_name)
                history_of_distances[class_name] = []

            src_emb = np.load(data_path)
            history_of_distances[class_name].append(self.clustering.calc_cosine_similarity(src_emb, target_emb))


        history_of_mean_distance_for_each_class = {}
        for class_name, distances in history_of_distances.items():
            history_of_mean_distance_for_each_class[class_name] = sum(distances) / len(distances)

        del history_of_distances
        target_person = max(history_of_mean_distance_for_each_class, key=history_of_mean_distance_for_each_class.get)

        print(target_person)

if __name__ == "__main__":
    face_retriever = FaceRetrieval()

    face_retriever.do_preprocessing()
    path_to_target_img = "/mnt/829A20D99A20CB8B/projects/github_projects/Face_Retrieval/test_imgs/TEST_BRAD_PIT.png"

    face_retriever.retrieve_face(path_to_face=path_to_target_img)