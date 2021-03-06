import os

config = {
    "gallery_path": os.path.join(os.getcwd(), "gallery"),
    "path_to_save_embs": os.path.join(os.getcwd(), "generated_embs"),
    "cpu_mode": True,
    "magface_pretrained_path": os.path.join(os.getcwd(), "pretrained_models", "magface_epoch_00025.pth"),
    "medoids_storage_path": os.path.join(os.getcwd(), "kmedoids"),
}


