from glob import glob 
import os
from sklearn_extra.cluster import KMedoids
import numpy as np
from config import config

class Clustering:
    def __init__(self, k=3, type="kmedoids", save_mode="True"):
        self.k = k
        self.type = type
        self.save_mode = save_mode

        if type == "kmedoids":
            self.KMedoids = KMedoids(n_clusters=self.k, random_state=0)

    def get_medoids(self, data_for_one_class):
        return self.KMedoids.fit(data_for_one_class)

    @staticmethod
    def save_medoid(medoid, path_to_save):
        np.save(medoid, path_to_save)


class Reader:
    def __init__(self):
        """
        * This class is to encapsulate all the utils which are useful (now or later) for reading data
        * No matter what is the type of data the parent path should satisfy the following structure:
        -Parent/Path/To/Data
            -Class A
            -Class B
            -Class C
            ...
        """
        pass
            
    def read_images_path(self, parent_path):
        classes = []
        images_path = []

        for img_file in sorted(glob(os.path.join(parent_path, "*", "*"))):
            classes.append(img_file.split("/")[-2])
            images_path.append(img_file)


        return zip(classes, images_path)

        

if __name__ == "__main__":

    clustering = Clustering()
    
    reader = Reader()
    path_data = reader.read_images_path(config["gallery_path"])
    
    for class_name, img_path in path_data:
        cv2.imread(img_path)

        

