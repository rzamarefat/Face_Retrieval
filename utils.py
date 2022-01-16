from glob import glob 
import os
from sklearn_extra.cluster import KMedoids
import numpy as np
from config import config
import cv2
from scipy import spatial
import shutil as shu

class Clustering:
    def __init__(self, k, type="kmedoids", save_mode="True"):
        self.k = k
        self.type = type
        self.save_mode = save_mode

        if type == "kmedoids":
            self.KMedoids = KMedoids(n_clusters=self.k, random_state=0)

    def get_medoids(self, data_for_one_class):
        kmed = KMedoids(n_clusters=self.k, metric='euclidean', method='alternate', init='heuristic', max_iter=300, random_state=4)
        kmed.fit(data_for_one_class)
        
        return kmed.cluster_centers_

    @staticmethod
    def save_medoid(medoid, path_to_save):
        np.save(path_to_save, medoid)


    def calc_cosine_similarity(self, src_emb, target_emb):
        return 1 - spatial.distance.cosine(src_emb, target_emb)



class Reader:
    def __init__(self):
        """
        * This class is to encapsulate all the utils which are useful (now or later) for reading data
        * No matter what is the type of data the parent path to the data should satisfy the following structure:
        -Parent/Path/To/Data
            -Class A
            -Class B
            -Class C
            ...
        """
        pass
            
    def read_path(self, parent_path):
        classes = []
        data_path = []

        for data in sorted(glob(os.path.join(parent_path, "*", "*"))):
            classes.append(data.split("/")[-2])
            data_path.append(data)


        return zip(classes, data_path)

    def read_npy(self, abs_path):
        with open(abs_path, "rb") as handle:
            data = np.load(handle, allow_pickle=True)
        
        return data


    def check_and_make_path(self, parent_folder, folder_to_make):
        path_to_make = os.path.join(parent_folder, folder_to_make)

        
        if os.path.isdir(path_to_make) and len(os.listdir(path_to_make)) == 0:  
            shu.rmtree(path_to_make, ignore_errors=True)

        os.mkdir(path_to_make)


class Preprocess:
    def __init__(self) -> None:
        self.target_size_for_magface = 112, 112

    def resize_img(self, img):
        return cv2.resize(img, self.target_size_for_magface)

    def show_img(self, img):
        cv2.imshow("Image", img)
        cv2.waitKey(0)



if __name__ == "__main__":

    clustering = Clustering(k=3)
    
    reader = Reader()
    path_to_embs = reader.read_path(config["path_to_save_embs"])
    

    obtained_classes = []
    data_to_feed_clustering = {}
    for class_name, data_path in path_to_embs:
        if not(class_name in obtained_classes):
            obtained_classes.append(class_name)
            data_to_feed_clustering[class_name] = []
        else:
            emb = reader.read_npy(data_path)
            # emb = emb[np.newaxis, :]
            data_to_feed_clustering[class_name].append(emb)
    
        
    for class_name, embs in data_to_feed_clustering.items():
        medoids = clustering.get_medoids(np.array(embs))
        
        if not(os.path.isdir(os.path.join(config["medoids_storage_path"], class_name))): 
            os.mkdir(os.path.join(config["medoids_storage_path"], class_name))
        
        
        for index, m in enumerate(medoids):
            abs_path_to_save = os.path.join(config["medoids_storage_path"], class_name, f"{index}.npy")
            clustering.save_medoid(m, abs_path_to_save)


    src_emb = np.load("/mnt/829A20D99A20CB8B/projects/github_projects/Face_Retrieval/kmedoids/A/0.npy")
    target_emb = np.load("/mnt/829A20D99A20CB8B/projects/github_projects/Face_Retrieval/kmedoids/B/2.npy")

    sim = clustering.calc_cosine_similarity(src_emb, target_emb)
    print(sim)