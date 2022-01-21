# Face Retrieval Pipeline

#### This is an implementation for retrieving face images of a given person from a given gallery.

**Bird's Eye View**
Below, you can see the whole pipeline of this retrieval system
![Screenshot from 2022-01-08 16-38-55](https://user-images.githubusercontent.com/79300456/148645364-1e6bb06d-b252-41c2-a0e4-6abaa00281e3.png)

**Explanation**  
In this section I am going to explain all the steps within this pipeline. Firstly, the embeddings of face images in the gallery databse is extracted with the help of MagFace which is a robust Face embedding generator. Then, the K-medoids of these embeddings are found and saved to the database of the medoids. When a query face image is fed to the pipeline, the embedding of this face image is extracted, and after that the similarity measure of the query embedding with all the K-Medoids of each class within our gallery is calculated using COSINE Similarity. Then, the top class with the most average of similarity is recognised. Finally, the similarity module is run on all the data points within the recognised class with the aim of finding the most to least similar images of the recognised person.
**Usage**  
In order to use this repo, you need to take the following steps:  
0 - Step zero is to download the pretrained model of MagFace(Reference is provided below) and put it inside a folder named "pretrained_models". Please note that this folder must be adjacent to Face_Retrieval module.The link to download:  
[MagFace Pretrained Model](https://drive.google.com/file/d/1Bd87admxOZvbIOAyTkGEntsEz3fyMt7H/view?usp=sharing)  
1- Put the face images inside the directory named gallery. For each person a seperate folder must be created. For instance:

```
gallery
-Person_One
-Person_Two
-Person_Three
- ...
```

2- Then you need to make sure that adjacent to the Face_Retrieval module, there are the following folders:

```
-generated_embs
-kmedoids
```

3-Initialise a FaceRetrieval instance and call the "do_preprocessing" method of this class. This method must be called just once for the same gallery.

```
face_retriever = FaceRetrieval()
face_retriever.do_preprocessing()
```

4- set the absolute path to your target face image and pass it to the retireve_face method of Face_Retrieval class.

```
path_to_target_img = "/mnt/829A20D99A20CB8B/projects/github_projects/Face_Retrieval/test_imgs/TEST_BRAD_PIT.png"
face_retriever.retrieve_face(path_to_face=path_to_target_img)
```

**Todos**

- [ ] Debug No GPU mode
- [ ] Add a module for aligning the faces
- [ ] Do directory ceating process automatically

**References**  
 This project is heavily dependent on the following academic work:

```
@inproceedings{meng2021magface,
  title={{MagFace}: A universal representation for face recognition and quality assessment},
  author={Meng, Qiang and Zhao, Shichao and Huang, Zhida and Zhou, Feng},
  booktitle=CVPR,
  year=2021
}
```
