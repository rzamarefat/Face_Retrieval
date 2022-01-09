import sys
import os

from torch._C import device, dtype
sys.path.append(os.path.join(os.getcwd()))

import torch
from torchvision import transforms
import numpy as np
from collections import OrderedDict
import iresnet
from config import config
from uuid import uuid1

class MagFaceEmbeddingGenerator:
    def __init__(self, embedding_size=512):
        self.transforms = transforms.Compose([
                    transforms.Normalize(
                        mean=[0., 0., 0.],
                        std=[1., 1., 1.]),
        ])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        features_model = iresnet.iresnet100(
            pretrained=False,
            num_classes=embedding_size,
        )
        self.features_model = self.__load_dict_inf(config, features_model).to(self.device)

    def __load_dict_inf(self, config, model):
        if os.path.isfile(config["magface_pretrained_path"]):
            
            if config["cpu_mode"]:
                checkpoint = torch.load(config["magface_pretrained_path"], map_location=torch.device("cpu"))
            else:
                checkpoint = torch.load(config["magface_pretrained_path"])
            _state_dict = self.__clean_dict_inf(model, checkpoint['state_dict'])
            model_dict = model.state_dict()
            model_dict.update(_state_dict)
            model.load_state_dict(model_dict)
            del checkpoint
            del _state_dict
            print("=> Magface pretrained model is loaded successfully")
        else:
            sys.exit(f"=> No checkpoint found at: {config['magface_pretrained_path']}")
        return model


    def __clean_dict_inf(self, model, state_dict):
        _state_dict = OrderedDict()
        for k, v in state_dict.items():
            # # assert k[0:1] == 'features.module.'
            # new_k = 'features.'+'.'.join(k.split('.')[2:])
            new_k = '.'.join(k.split('.')[2:])
            if new_k in model.state_dict().keys() and \
            v.size() == model.state_dict()[new_k].size():
                _state_dict[new_k] = v
            # assert k[0:1] == 'module.features.'
            new_kk = '.'.join(k.split('.')[1:])
            if new_kk in model.state_dict().keys() and \
            v.size() == model.state_dict()[new_kk].size():
                _state_dict[new_kk] = v
        num_model = len(model.state_dict().keys())
        num_ckpt = len(_state_dict.keys())
        if num_model != num_ckpt:
            sys.exit("=> Not all weights loaded, model params: {}, loaded params: {}".format(
                num_model, num_ckpt))
        return _state_dict


    def __preprocess_data(self, img):
        img = np.array(img, dtype=np.float32)
        empty_tensor = np.zeros((img.shape[0], img.shape[1], img.shape[2]), dtype=np.float32)
        img = np.array([img, empty_tensor])

        img_tensor = self.transforms(torch.from_numpy(img).permute(0, 3, 1, 2))

        img_tensor = img_tensor.to(self.device)
        
        return img_tensor
        

    def generate_embeddings(self, img, abs_path_to_save_emb):
        if isinstance(type(img), np.ndarray):
            raise Exception("The type of data given to 'generate_embeddings' method is not a numpy array. \
You have probably forgotten to read the images as np.array")
        
        if not(img.shape[1] == 112 or img.shape[2] == 112):
            raise Exception("The size of images provided for this 'generate_embeddings' method must be 112*112")

        img = self.__preprocess_data(img)
        generated_embs = self.features_model(img)

        self.__save_emb(generated_embs[0].detach().cpu(), abs_path_to_save_emb)
        

    def __save_emb(self, emb, abs_path_to_save_emb):
        np.save(abs_path_to_save_emb.replace(abs_path_to_save_emb.split(".")[-1], "npy"), emb)

if __name__ == "__main__":
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


    
        

        
        