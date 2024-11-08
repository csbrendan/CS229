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

## Attribution

Part of the code for supervised-fine-tuning and dpo was adapted and inspired from a deeplearning.ai notebook, "Supervised fine-tuning (SFT) of an LLM" and "Human preference fine-tuning using direct preference optimization (DPO) of an LLM" by Lewis Tunstall and Edward Beeching of Hugging Face. 

This project utilizes the PubMed QA dataset.
https://huggingface.co/datasets/qiaojin/PubMedQA


## Run Experiments

To perform supervise fine-tuning:

python biomistral_qa_context_sft.py  -> creates SFT model (data/biomistral-7b-sft-pqa-context-lora)

biomistral_qa_context_sft_gen_answers.py -> generates answers to be used in DPO step: sft_pqa_context_inference_results.json

biomistral_qa_context_dpo.py -> creates DPO model (data/biomistral-7b-dpo-pqa-context-lora)

biomistral_dpo_context_gen_answers.py -> generates answers to be used in Eval: dpo_inference_pqa_context_results.json


