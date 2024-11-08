# CS229
# Term Project for CS229 Machine Learning by Andrew Ng

# 

# Enhancing Video Conference Deepfake Detection: Domain Adaptation of AltFreezing with Adversarial Training

## Paper
https://github.com/csbrendan/CS229/blob/main/paper/CS229_Project.pdf

## Poster
https://github.com/csbrendan/CS229/blob/main/poster/CS229_Project_Poster_Final.pdf

## Video
https://youtu.be/PYhsdQp6KhM

## Requirements

- Nvidia A100
- PyTorch
- FaceForensics++
- AltFreezing


# Facial Deepfake Detection

**Project Overview**

This project aims to detect deepfake videos using a state-of-the-art deep learning model. We leverage the FaceForensics dataset for training and evaluation.

**Dataset**

* **FaceForensics:** This dataset contains a large collection of real and fake videos, providing a rich source of training data.

## Attribution and Model Architecture

I utilize a deep neural network architecture, [AltFreezing](https://github.com/ZhendongWang6/AltFreezing), to extract relevant features from video frames. The model is fine-tuned on custom datasets for adaptation and to improve performance.



**Evaluation**

The model is evaluated on a held-out test set from the FaceForensics dataset, measuring its accuracy in detecting deepfake videos.

**Future Work**

* Explore advanced techniques like adversarial training to improve robustness against adversarial attacks.
* Investigate the use of attention mechanisms to focus on key regions of the face.
* Develop a real-time deepfake detection system for practical applications.

**Note:** For detailed usage and configuration options, refer to the specific scripts.







## Attribution

Part of the code for supervised-fine-tuning and dpo was adapted and inspired from a deeplearning.ai notebook, "Supervised fine-tuning (SFT) of an LLM" and "Human preference fine-tuning using direct preference optimization (DPO) of an LLM" by Lewis Tunstall and Edward Beeching of Hugging Face. 

This project utilizes the PubMed QA dataset.
https://huggingface.co/datasets/qiaojin/PubMedQA


## Run Experiments

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
     python ft_augment.py
     ```
   * Or, fine-tune the model on a custom dataset without augmentation:
     ```bash
     python finetune_custom_data.py
     ```


