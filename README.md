# Question Answering on SQuAD v2 using RoBERTa

This repository presents a complete, end-to-end extractive Question Answering (QA) system implemented on the SQuAD v2 dataset using RoBERTa and HuggingFace Transformers.

The project is designed to demonstrate an engineering-level understanding of QA systems by explicitly implementing preprocessing, training, inference, and evaluation logic instead of relying on high-level black-box pipelines.

---

## Project Overview

The system covers the full lifecycle of a modern QA model:

• Sliding-window tokenization to handle long contexts  
• Overflow-to-sample mapping to reconnect chunks with original questions  
• CLS-based no-answer detection required for SQuAD v2  
• Span-based answer extraction using start and end logits  
• Aggregation of multiple chunks into a single best answer per question  
• Evaluation using official Exact Match (EM) and F1 metrics  

All core components are implemented manually to expose how extractive QA models work internally.

---

## Repository Structure

qa-squadv2-roberta/  
train.py – Fine-tunes RoBERTa on SQuAD v2  
eval.py – Runs inference and computes EM/F1 using custom span selection  
requirements.txt  
.gitignore  

Trained model weights are intentionally excluded from this repository due to size constraints.

---

## Training

The model is fine-tuned using RoBERTa on a subset of the SQuAD v2 dataset.

Training configuration:

Model: deepset/roberta-base-squad2  
Dataset: SQuAD v2  
Training subset: 25,000 examples  
Validation subset: 5,000 examples  
Mixed-precision training (fp16 enabled)  

To start training, run:

python train.py  

The fine-tuned model is saved locally after training.

---

## Evaluation

Evaluation follows the official SQuAD v2 protocol.

The evaluation pipeline performs the following steps:

• Tokenizes validation data using sliding windows  
• Runs inference on all context chunks  
• Selects the best answer span per question using logits  
• Handles no-answer cases using CLS comparison  
• Computes Exact Match (EM) and F1 scores  

To run evaluation, execute:

python eval.py  

Example evaluation results:

Exact Match (EM): ~41  
F1 Score: ~41  
No-answer accuracy: ~81  

---

## Results Analysis

The model shows strong performance on no-answer detection and moderate performance on answerable questions. The primary source of error is answer boundary selection, which is a common challenge in extractive QA systems.

Performance can be further improved through span selection tuning, threshold adjustment for no-answer decisions, and extended fine-tuning.

---

## Engineering Focus

This project emphasizes:

• Explicit QA logic instead of abstract pipelines  
• Evaluation-driven development  
• Clear separation between training and evaluation scripts  
• Reproducible and extensible code structure  

The implementation reflects an ML engineering mindset rather than a tutorial-style approach.

---

## Future Improvements

• Threshold tuning for no-answer probability  
• Batched inference for faster evaluation  
• Model comparison with lighter QA architectures  
• Detailed error analysis and visualization
