# Ascertainment of clinical outcomes and metastatic sites from radiology impression text using large language models
Clinical outcomes ascertainment and sites of metastases ascertainment using large language models (LLMs). Dataset is de-identified radiology impressions with manually annotated labels. 

## Model Experiments

### GPT-4 & GPT-4o Models  
The same scripts were used for both **GPT-4** and **GPT-4o**, with only the model name changed.

#### Zero-Shot  
- `/code/gpt_models/code_zeroshot.ipynb`

#### One-Shot  
- `/code/gpt_models/code_oneshot.ipynb`

---

### Open-Source Models  
#### Zero-Shot & One-Shot  
- **MAMBA Model**  
  - `/code/opensource_models/zero-one_shot/run_mamba_model_log_prob.py`  
  - `/code/opensource_models/zero-one_shot/run_mamba_model_log_prob.sh`  
- **Other Open-Source Models (BioMistral-7.3B, Mistral0.2-7.3B, Mistral0.3-7.3B, LLaMa2-6.7B, LLaMa3.1-8B)**  
  - `/code/opensource_models/zero-one_shot/run_opensource_models_log_prob.py`  
  - `/code/opensource_models/zero-one_shot/run_opensource_models_log_prob.sh`  

---

#### Supervised Fine-Tuning (SFT)  
These scripts are used for fine-tuning open-source models.  
- `/code/opensource_models/sft/run.sh`  
- `/code/opensource_models/sft/run_causal_models.py`  

---

### Evaluation  
This Jupyter notebook is used for evaluating model performance.  
- `/code/opensource_models/evaluation/eval_notebook.ipynb`  
