# Scribe Verification Using Siamese, Triplet, and Vision Transformer Networks

This repository contains a deep metric learning framework for **scribe verification** on ancient and modern Chinese handwriting images.  
The objective is to automatically determine whether two manuscript fragments were written by the same scribe by learning discriminative embeddings using **Siamese**, **Triplet**, and **ViT-based** neural architectures.

The project includes:
- Custom MobileNetV3-based Siamese model (MobileNetV3+)
- Triplet-learning architecture with dynamic sampling
- ViT-B/16 Siamese implementation
- Unified PyTorch training & evaluation framework
- Contrastive and Triplet loss functions
- Fixed-pair evaluation and ROC/AUC metric computation

---

## 📁 Project Structure

```
├── configs/                            # YAML configuration files
├── data/                               # (Not included) Dataset folders: train/ and test/
├── src/
│   ├── models/                         # CNN, Siamese, Triplet, ViT architectures
│   ├── dataset_pairs.py                # Pair dataset for Siamese training
│   ├── dataset_triplets.py             # Triplet dataset for Triplet learning
│   ├── transforms.py                   # Image preprocessing pipeline
│   ├── losses.py                       # Contrastive and Triplet losses
│   ├── metrics.py                      # ROC/AUC/FAR/FRR metrics
│   └── viz.py                          # ROC visualization# utilities
├── train_siamese.py                    # Siamese training script
├── train_triplet.py                    # Triplet training script
├── evaluate_siamese.py                 # Evaluation on fixed pairs
├── evaluate_siamese_perclass.py        # Evaluation on fixed pairs - ROC per class, AUC/ACC/FAR/FRR graphs and FN/FP Confusion Matrices
├── evaluate_siamese_perclass_full.py   # Evaluation on fixed pairs - ROC per class, AUC/ACC/FAR/FRR graphs, FN/FP Confusion Matrices and Confusion Matrices TP/TN/FP/FN per class
├── evaluate_triplet.py                 # Evaluation on triplet
├── evaluate_siamese_perclass.py        # Evaluation on triplet - ROC per class, AUC/ACC/FAR/FRR graphs and FN/FP Confusion Matrices
├── evaluate_siamese_perclass_full.py   # Evaluation on triplet - ROC per class, AUC/ACC/FAR/FRR graphs, FN/FP Confusion Matrices and Confusion Matrices TP/TN/FP/FN per class
└── README.md               # Project documentation
```

---

## 📊 Datasets Used

### **1. Tsinghua Bamboo Slips Dataset**
Ancient Chinese manuscript fragments attributed to different scribes.  
Used for experiments on historical scribe verification.

### **2. MCCD: Multi-Attribute Chinese Calligraphy Dataset**
A modern dataset containing handwritten Chinese characters from multiple calligraphers.  
A subset of writers with the largest number of samples was used to ensure balanced training.

---

## 📁 Dataset Availability

Both datasets used in this project are **not openly downloadable**, but:

> **They can be accessed upon reasonable request to the original authors for research purposes.**

- The **Tsinghua Bamboo Slips Dataset** can be requested from the authors of the corresponding publication.  
- The **MCCD dataset** is available upon request according to the instructions in its original paper.

This repository **does not include any dataset files**, and only publicly permitted samples were used.

---

## ⚙️ Installation

```bash
git clone <repository-url>
cd ScribeVerification
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🏋️ Training

### Train a Siamese Network
```bash
python train_siamese.py --config configs/default.yaml
```

### Train a Triplet Network
```bash
python train_triplet.py --config configs/triplet.yaml
```

---

## 🧪 Evaluation

### Evaluate a Siamese Model
```bash
python evaluate_siamese_perclass_full.py `
 --model_module src.models.<script> `
 --ckpt checkpoints\<model>_<architecture>_e<epoch_number>.pt `
 --out_dir eval_<model>_<architecture>
```

### Evaluate a Triplet Model
```bash
python evaluate_triplet_perclass_full.py `
 --model_module src.models.<script> `
 --ckpt checkpoints\<model>_<architecture>_e<epoch_number>.pt `
 --out_dir eval_<model>_<architecture>
```

## 📄 Evaluation based on test_pairs.csv

All evaluations are driven by a fixed CSV file:

```
path1,path2,label
Bao_Xun/img_001.png,Bao_Xun/img_014.png,1
Bao_Xun/img_003.png,Guan_Zhong/img_008.png,0
...
```

- `label = 1`: same scribe
- `label = 0`: different scribes

This ensures **reproducible and fair evaluation**.

---

## 🔍 Example: Why Bao Xun Has 4106 Evaluated Pairs

Although Bao Xun has ~40 test images:

- Positive pairs: C(40,2) = 780
- Negative pairs: Bao Xun images paired with many other scribes ≈ 3300+

Total ≈ **4106 evaluation pairs**

This is expected in **metric learning verification**.

---

---

## 📈 Results Summary

- **MobileNetV3+ Custom (Siamese)** achieved the best overall performance  
  - AUC: ~0.958  
  - Accuracy: ~89.1%  
- Triplet learning did **not** outperform contrastive learning in this domain.  
- ViT-B/16 Siamese performed lower than CNN-based methods on both datasets.

---

## 📬 Contact

For dataset access, please contact the original dataset authors as listed in their respective publications.

For questions regarding this repository, feel free to open an issue or reach out via email.

---

## 📄 License

This project is released for academic and research use.  
Redistribution of dataset files is strictly prohibited.
