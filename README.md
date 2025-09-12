Code Portfolio

This repository highlights some of my example works in **AI/ML, Bioinformatics, and Computational Biology** through practical code examples. The projects cover **data analysis, web generative AI, predictive modeling, omics analysis, and structural bioinformatics**, reflecting my research contributions and technical expertise.

---

👩‍💻 About Me
Sayane Shome – Computational researcher passionate about AI in healthcare and life sciences.

Website: www.sayaneshome1.com

LinkedIn: linkedin.com/in/sayaneshome

GitHub: github.com/sayaneshome


## 📂 Repository Structure
```
Code_Portfolio_sayaneshome1/
├── notebooks/
│   ├── Notebook01.ipynb -- Single-cell RNA seq analysis of downstream analysis of counts generated from 10x cellranger using Scanpy.
│   ├── Notebook02.ipynb -- Single-cell RNA seq analysis particularly RNA velocity,pseudo-velocity computation etc.
│   ├── Notebook03.ipynb -- Bulk-RNA seq analysis, particularly focusing on differential expression analysis based on two genotypes/treatments and interaction terms ,and additional downstream analysis
│   ├── Notebook04.ipynb -- Hi-C analysis, which helps find TADs,interaction loops specially comparing between wild-type/knockout samples.
│   ├── Notebook05.ipynb
│   ├── 06_.ipynb
│   ├── 07_.ipynb
│   ├── 08_.ipynb
├── datasets/          # optional sample data
├── images/            # visualizations/screenshots
├── environment.yml    # environment dependencies
└── README.md
```



## 🔍 Overview of Projects

### **1. Generative AI for Medical Imaging**
- **Notebook:** <will be added>.
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
- **Notebook:** `Sc-RNAseq_example dataset IKKB`
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

### **5. Chatbot using LLMs providing details about clinical trials in a conversational manner
Data Analysis**
- **Notebook:** `05_HiC_Data_Analysis.ipynb`
- **Highlights:**  
  - Analyze `.cool` and `.mcool` Hi-C files for chromatin structure.  
  - Compare WT vs KO samples.  
- **Tech Stack:** Python, OpenAI-LLM, Streamlit.

## ⚙️ Setup
```bash
git clone https://github.com/sayaneshome/Code_Portfolio_sayaneshome1.git
cd Code_Portfolio_sayaneshome1
conda env create -f environment.yml
conda activate code-portfolio
jupyter notebook

