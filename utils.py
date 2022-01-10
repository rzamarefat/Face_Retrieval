from glob import glob 
import os
from sklearn_extra.cluster import KMedoids
import numpy as np
from config import config
import cv2

class Clustering:
    def __init__(self, k=3, type="kmedoids", save_mode="True"):
        self.k = k
        self.type = type
        self.save_mode = save_mode

        if type == "kmedoids":
            self.KMedoids = KMedoids(n_clusters=self.k, random_state=0)

    def get_medoids(self, data_for_one_class):
        kmedoids = self.KMedoids.fit(data_for_one_class)
        return kmedoids.labels_

    @staticmethod
    def save_medoid(medoid, path_to_save):
        np.save(medoid, path_to_save)


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
        print(medoids)







        

