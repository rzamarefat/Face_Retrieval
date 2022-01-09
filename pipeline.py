from MagFaceEmbeddingGenerator import MagFaceEmbeddingGenerator
import config
import os
import cv2

magface_embedding_generator = MagFaceEmbeddingGenerator()

for person in sorted(os.listdir(config["gallery_path"])):
    if not(os.path.isdir(os.path.join(config["path_to_save_embs"], person))): 
        os.mkdir(os.path.join(config["path_to_save_embs"], person))

    for img_file in sorted(os.listdir(os.path.join(config["gallery_path"], person))):
        abs_path_to_img = os.path.join(config["gallery_path"], person, img_file)
        abs_path_to_save = os.path.join(config["path_to_save_embs"], person, img_file)

        img = cv2.imread(abs_path_to_img)
        img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        magface_embedding_generator.generate_embeddings(img, abs_path_to_save)


if __name__ == "__main__":
    pass