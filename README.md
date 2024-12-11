# CS229 Machine Learning with Andrew Ng


# Detecting Deepfakes in Video Conferencing Scenarios: A Study on Domain-Specific Challenges and Model Adaptations

## Paper
https://github.com/csbrendan/CS229/blob/main/paper/CS229_PROJ_FINAL.pdf

## Poster
https://github.com/csbrendan/CS229/blob/main/poster/CS229_Project_Poster.pdf

## Video
[https://www.youtube.com/watch?v=rNlEeR-ge5A](https://www.youtube.com/watch?v=rNlEeR-ge5A)

## Requirements

- Nvidia A100
- PyTorch
- FaceForensics++
- Celebrity Deepfake
- DeepLiveCam
- AltFreezing



## Project Overview ##

This project aims to detect deepfake videos in video conferences using a customized SOTA deep learning model. I leverage my own dataset and the FaceForensics and Celeb-DF datasets for training and evaluation.



## Attribution and Model Architecture

**Model**

I utilize a deep neural network architecture, [AltFreezing](https://github.com/ZhendongWang6/AltFreezing), to extract relevant features from video frames. The model is fine-tuned on custom datasets for adaptation and to improve performance.

* **DeepLiveCam:** This is a real-time face swapping tool used to generate manipulated video content: [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)

**Dataset**

* **FaceForensics:** This dataset contains a large collection of real and fake videos, providing a rich source of training data: [FaceForensics++](https://github.com/ondyari/FaceForensics)


* **Celeb-DF:** This dataset consists of high-quality celebrity deepfake videos and their original counterparts: [Celeb-DeepfakeForensics](https://github.com/yuezunli/celeb-deepfakeforensics)


**Evaluation**

The model is evaluated on a held-out test set from my custom Zoom dataset manipulated with Deep Live Cam.





