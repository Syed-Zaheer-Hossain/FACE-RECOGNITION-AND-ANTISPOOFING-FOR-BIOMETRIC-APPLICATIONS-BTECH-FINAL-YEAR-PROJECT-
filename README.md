# Face Recognition and Anti-Spoofing for Biometric Applications  
### B.Tech Final Year Project

This repository contains the implementation of a **Face Recognition and Anti-Spoofing System** designed for secure biometric authentication.  
The system integrates **face recognition** with **liveness (anti-spoofing) detection** to prevent presentation attacks such as printed photos, mobile screen replays, and video attacks.

This work is supported by a **self-made dataset** and has been **published as a research paper**.

---

## 📌 Project Highlights

- Real-time face recognition  
- Anti-spoofing (liveness detection)  
- Self-made dataset (real + spoof samples)  
- Integrated GUI application  
- Published research work (Springer)  

---

## 📚 Published Research Paper

**FRAD: A Face Recognition and Anti-spoofing Database for Biometric Applications**

- **DOI:** https://doi.org/10.1007/978-981-96-7614-9_32  
- **Publisher:** Springer (Lecture Notes in Networks and Systems (LNNS,volume 1358))
- **Authors:**  Saiyed Umer, Surjit Mandal, Syed Zaheer Hossain, Anirban Guha, Ranjeet Kumar Rout  

If you use this project or dataset for academic or research purposes, please cite this paper.

---

## 🚀 Project Overview

Biometric systems based on facial recognition are widely used in access control, attendance systems, and identity verification.  
However, traditional face recognition systems are vulnerable to **spoofing attacks** such as:

- Printed photographs  
- Mobile/laptop screen replays  
- Video-based attacks  

This project addresses these challenges by combining:
- **Face Recognition** for identity verification  
- **Anti-Spoofing Detection** to ensure the presence of a live human subject  

Authentication is granted **only when both modules succeed**, making the system robust and secure.

---

## 🧠 Key Features

- Face detection and preprocessing  
- Facial feature extraction  
- Anti-spoofing (liveness) classification  
- Real-time webcam input  
- GUI-based user interaction  
- Modular and extensible codebase  

---

```
FACE-RECOGNITION-AND-ANTISPOOFING-FOR-BIOMETRIC-APPLICATIONS-BTECH-FINAL-YEAR-PROJECT
│
├── ANTI SPOOFING/
│   ├── anti-spoof-facenet.py
│   ├── anti-spoof-lbp.py
│   ├── anti_spoof_hog.py
│   ├── antispoofing_vgg16.py
│   ├── antispoofing_vgg19.py
│   └── dense_antispoof.py
│
├── FACE PREPROCESSING/
│   ├── Recognition_Face_crop_Final_MTCNN.py
│   └── anti-spoof_Face_crop_Final_MTCNN.py
│
├── FACIAL FEATURE EXTRACTION/
│   ├── Face_Recognition_Facenet.ipynb
│   ├── feature_extraction_FaceNet.py
│   ├── feature_extraction_hog_final.py
│   ├── lbp-final.py
│   ├── vgg16.py
│   └── vgg19.py
│
├── FINAL COMBINED GUI INTERFACE TKINTER/
│   ├── Final gui interface combined recognition antispoofing.py
│   └── Recognition_GUI.ipynb
│
└── README.md
```

---

## 🗂️ Folder Description Summary

| Folder / File | Description |
|--------------|-------------|
| **AntiSpoofing/** | Liveness detection and spoof prevention modules |
| **FaceRecognition/** | Face detection, feature extraction, and identity recognition |
| **GUI_Interface/** | Integrated real-time GUI application |
| **README.md** | Project documentation |

---

## 📊 Dataset Description

- The dataset is **self-created** for this project  
- Contains both **real (live)** and **spoof (fake)** samples  
- Spoof samples include:
  - Printed photographs  
  - Screen replay attacks  
  - Video-based attacks  
- Used for training and evaluating both face recognition and anti-spoofing models  

---

## 🛠️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/Syed-Zaheer-Hossain/FACE-RECOGNITION-AND-ANTISPOOFING-FOR-BIOMETRIC-APPLICATIONS___BTECH-FINAL-YEAR-PROJECT.git
cd FACE-RECOGNITION-AND-ANTISPOOFING-FOR-BIOMETRIC-APPLICATIONS___BTECH-FINAL-YEAR-PROJECT
```

### 2️⃣ Create Virtual Environment (Optional)
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Running the Application

```bash
python main.py
```

---

## 🔍 System Workflow

1. Capture face input using webcam  
2. Detect and preprocess face region  
3. Perform face recognition  
4. Apply anti-spoofing (liveness detection)  
5. Authenticate user only if both checks pass  

---

## 🧪 Future Enhancements

- Deep learning-based anti-spoofing models  
- Integration with FaceNet / ArcFace  
- Mobile or embedded deployment  
- Encrypted biometric storage  
- Cloud-based authentication  

---

## 🤝 Acknowledgements

- Springer LNCS
- Academic mentors and faculty (Aliah University, Kolkata, West Bengal, India)
- Contributors and reviewers
