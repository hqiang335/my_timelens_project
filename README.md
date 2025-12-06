# Event-based Video Frame Interpolation with TimeLens-XL and U-Net Refiner Custom Dataset Support + Refiner Training Pipeline

This repository extends **TimeLens-XL (ECCV 2024)** with:

- Support for our **self-recorded MyData dataset**
- Full compatibility with the **HQEVFI dataset**
- Optional **U-Net Refiner module** for improving interpolation quality
- A structured **freeze–refine–finetune** training workflow

---

## 📦 Dataset

### **1. MyData (Our Self-recorded Event Camera Dataset)**

APS RGB frames + Event `.npy` files recorded using a DAVIS-style sensor.

👉 **Download (MyData dataset):**  
https://hkustgz-my.sharepoint.com/:u:/g/personal/hqiang669_connect_hkust-gz_edu_cn/EQytdQ-Rg09Pv0riEXI3vIABFVr8QX5Ib5lLau6oqRd-cw?e=14tgds

Place it under:

```
mydata/
   scene1/
      aps_png/
      events/
```

---

### **2. HQEVFI Dataset (Official TimeLens-XL Dataset)**

👉 Download link (Google Drive):  
https://drive.google.com/file/d/104ZMJ-M_frImOOCGfLk_HDb2FV1trveT/view?usp=drive_link

---

## 🎥 Demo Video

👉 **Demo Video Link:**  
https://hkustgz-my.sharepoint.com/:f:/g/personal/hqiang669_connect_hkust-gz_edu_cn/Ekw0jvvcFGZAhdTp-j4rR4wB7UnQroevsLyvKcGaCELYBg?e=ZhQh9X

---

## 🧰 Pretrained Weights

### **1. Official TimeLens-XL Pretrained Model (no Refiner)**

Used for:

- Baseline inference on HQEVFI  
- Frozen-backbone Refiner training  

```
weights/Expv8_large_HQEVFI.pt
```

---

# ⚡ Quick Start

## 0. Install environment

```bash
pip install -r requirements.txt
```

---

# **1. HQEVFI — Inference using official pretrained model (interp ratio = 4)**

```bash
python run_network.py   --param_name traintest_RC_smallmix_lpips   --model_name Expv8_large   --model_pretrained weights/Expv8_large_HQEVFI.pt   --skip_training   --extension _pretrained_inference_x4
```

---

# **2. HQEVFI — Training using official pretrained model (interp ratio = 4)**

Just remove `--skip_training`:

```bash
python run_network.py   --param_name traintest_RC_smallmix_lpips   --model_name Expv8_large   --model_pretrained weights/Expv8_large_HQEVFI.pt   --extension _pretrained_train_x4
```

---

# **3. HQEVFI — Freeze Backbone, Train Only the U-Net Refiner**

```bash
python run_network_refine.py   --param_name traintest_RC_smallmix_lpips   --model_name Expv8_large   --model_pretrained ./weights/Expv8_large_HQEVFI.pt   --extension _unetRefiner
```

This stage:

- Freezes the TimeLens-XL backbone  
- Trains only the `unet_refiner.*` parameters  

---

# **4. HQEVFI — After Refiner Training, Finetune the Whole Model**

Replace the pretrained path with the newly produced checkpoint:

```bash
python run_network_refine.py   --param_name traintest_RC_smallmix_lpips   --model_name Expv8_large   --model_pretrained <path_to_trained_refiner_weight.pt>   --extension _unetRefiner_finetune
```

---

# **5. Adjusting Training Epochs**

The official pretrained model `Expv8_large_HQEVFI.pt` contains:

```
epoch = 10
```

If you want to train **5 more epochs**, set:

File:  
```
params/HQEVFI/params_traintest_quick_adam_withlpips_mix.py
```

Modify:

```python
training_config.max_epoch = 15  # 10 original + 5 new epochs
```

---

# **6. Inference on MyData Dataset (interp ratio = 4)**

```bash
python run_network.py   --param_name MyInferenceDataset   --model_name Expv8_large_infer   --model_pretrained <your_weight_path>   --skip_training   --extension _x4
```

---

## 📁 Project Structure (Simplified)

```
my_timelens_project/
│
├── dataset/
├── params/
│   ├── HQEVFI/
│   ├── mydata/
│
├── weights/
│     └── Expv8_large_HQEVFI.pt
│
├── mydata/     # our custom dataset
├── run_network.py
├── run_network_refine.py
└── generate_video.py
```

---

## 🙏 Acknowledgements

This project is based on:

**TimeLens-XL (ECCV 2024)**  
Project page:  
https://openimaginglab.github.io/TimeLens-XL/

We extend the original framework with:

- MyData dataset support  
- U-Net Refiner integration  
- Freeze–Refine–Finetune training pipeline  
