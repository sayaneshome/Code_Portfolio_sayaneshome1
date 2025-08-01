Code Portfolio (Work in progress; will complete by Aug 3,2025)

This repository highlights some of my example works in **AI/ML, Bioinformatics, and Computational Biology** through practical code examples. The projects cover **generative AI, predictive modeling, omics analysis, and structural bioinformatics**, reflecting my research contributions and technical expertise.

---

👩‍💻 About Me
Sayane Shome – Computational researcher passionate about AI in healthcare and life sciences.

Website: www.sayaneshome1.com

LinkedIn: linkedin.com/in/sayaneshome

GitHub: github.com/sayaneshome

⭐ If you find this helpful, consider giving the repo a star!

## 📂 Repository Structure
Code_Portfolio_sayaneshome1/
│
├── notebooks/
│ ├── 01_Generative_AI_Medical_Imaging.ipynb
│ ├── 02_
│ ├── 03_Bulk_RNAseq_Analysis.ipynb
│ ├── 04_SingleCell_RNAseq_Analysis.ipynb
│ ├── 05_HiC_Data_Analysis.ipynb
│ ├── 06_Molecular_Dynamics_Simulation.ipynb
│ ├── 07_
│ ├── 08_
│
├── datasets/ (optional sample data)
├── images/ (visualizations/screenshots)
├── environment.yml
└── README.md


---

## 🔍 Overview of Projects

### **1. Generative AI for Medical Imaging**
- **Notebook:** `01_Generative_AI_Medical_Imaging.ipynb`
- **Highlights:**  
  - Text-to-image synthesis using diffusion models for neonatal X-rays.  
  - BLIP-2 integration for image captioning.  
  - Metrics: SSIM, FID, BERTScore.  
- **Tech Stack:** PyTorch, Hugging Face Diffusers, CLIP, BLIP-2.

---

### **2. Predictive Modeling Using EHR**
- **Notebook:** `02_EHR_Predictive_Modeling.ipynb`
- **Highlights:**  
  - Predict BPD/RDS outcomes and Length of Stay using EHR.  
  - Models: XGBoost, Random Forest.  
  - Interpretability with SHAP.  
- **Tech Stack:** Python, scikit-learn, pandas.

---

### **3. Bulk RNA-seq Analysis**
- **Notebook:** `03_Bulk_RNAseq_Analysis.ipynb`
- **Highlights:**  
  - Preprocessing (QC, normalization).  
  - Differential expression analysis using DESeq2.  
  - Visualization: volcano plots, PCA.  
- **Tech Stack:** R/Python (DESeq2, pandas, matplotlib).

---

### **4. Single-cell RNA-seq Analysis**
- **Notebook:** `04_SingleCell_RNAseq_Analysis.ipynb`
- **Highlights:**  
  - Preprocessing and normalization of scRNA-seq.  
  - Dimensionality reduction (PCA, UMAP), clustering.  
  - Cell-type annotation and marker gene detection.  
- **Tech Stack:** Scanpy, Seurat, Python.

---

### **5. Hi-C Data Analysis**
- **Notebook:** `05_HiC_Data_Analysis.ipynb`
- **Highlights:**  
  - Analyze `.cool` and `.mcool` Hi-C files for chromatin structure.  
  - Compare WT vs KO samples.  
- **Tech Stack:** Python, Cooltools.

---

### **6. Molecular Dynamics Simulation**
- **Notebook:** `06_Molecular_Dynamics_Simulation.ipynb`
- **Highlights:**  
  - Simulate membrane transport proteins.  
  - Analyze transport pathways and flexibility.  
- **Tech Stack:** NAMD, VMD, Python.

---

### **7. Non-coding RNA Annotation**
- **Notebook:** `07_ncRNA_Annotation.ipynb`
- **Highlights:**  
  - Pipeline for ncRNA identification and annotation from RNA-seq data.  
- **Tech Stack:** Python, Bioconductor.

---

### **8. Protein Function Prediction**
- **Notebook:** `08_Protein_Function_Prediction.ipynb`
- **Highlights:**  
  - Predict protein function using ML based on sequence and structure features.  
- **Tech Stack:** Python, scikit-learn, Biopython.

---

## ⚙️ Setup
```bash
git clone https://github.com/sayaneshome/Code_Portfolio_sayaneshome1.git
cd Code_Portfolio_sayaneshome1
conda env create -f environment.yml
conda activate code-portfolio
jupyter notebook

