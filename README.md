# CS229 Machine Learning with Andrew Ng


# Adapting Deepfake Detection to Video Conferencing: Domain-Specific Augmentation and Targeted Model Fine-tuning

## Paper
https://github.com/csbrendan/CS229/blob/main/paper/CS229_PROJ_FINAL.pdf

## Poster
https://github.com/csbrendan/CS229/blob/main/poster/CS229_Project_Poster_Final.pdf

## Video
https://youtu.be/PYhsdQp6KhM

## Requirements

- Nvidia A100
- PyTorch
- FaceForensics++
- Celebrity Deepfake
- AltFreezing
- Deep Live Cam



## Project Overview ##

This project aims to detect deepfake videos in video conferences using a customized SOTA deep learning model. I leverage my own dataset and the FaceForensics dataset for training and evaluation.



## Attribution and Model Architecture

**Model**

I utilize a deep neural network architecture, [AltFreezing](https://github.com/ZhendongWang6/AltFreezing), to extract relevant features from video frames. The model is fine-tuned on custom datasets for adaptation and to improve performance.


**Dataset**

* **FaceForensics:** This dataset contains a large collection of real and fake videos, providing a rich source of training data: [FaceForensics++](https://github.com/ondyari/FaceForensics)

**Evaluation**

The model is evaluated on a held-out test set from the FaceForensics dataset and my custom dataset, measuring its accuracy in detecting deepfake videos.


## To Run Experiments

To perform supervise fine-tuning:

1. **Data Preparation:**
2. 
   * Download and process the FaceForensics dataset using the provided script:
     ```bash
     python faceforensics_download_v4.py
     ```
3. **Model Training:**
   * Fine-tune the model on a custom augmented dataset:
     ```bash
     python finetune_augment.py
     ```
   * Or, fine-tune the model on a custom dataset without augmentation:
     ```bash
     python finetune_custom_data.py
     ```


**Future Work**

* Explore advanced techniques like adversarial training to improve robustness against adversarial attacks.
* Investigate the use of attention mechanisms to focus on key regions of the face.
* Integrate explainability via visualizations such as heatmaps.
