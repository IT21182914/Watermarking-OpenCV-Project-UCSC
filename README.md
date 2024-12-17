# **Watermarking Project 🖼️**

This project implements **digital image watermarking** techniques to embed and detect watermarks in images, ensuring robustness against noise and modifications.

---

## **Features 🚀**  
### 1. **Dataset Splitting**  
- Divides 13,000 images into three manageable sets.

### 2. **Blind Watermarking**  
- Embeds 1-bit watermarks (0 and 1) into images.

### 3. **Noise Addition**  
- Simulates external damage using:  
   - **Gaussian Noise**  
   - **Shot Noise**  
   - **Salt & Pepper Noise**  

### 4. **Watermark Detection**  
- Detects hidden watermarks using correlation analysis.

### 5. **Informed Watermarking**  
- Strengthens watermark embedding with Gaussian-based patterns using an alpha value.

---

## **How to Run 🛠️**  
1. **Clone the repository:**  
   ```bash
   git clone [https://github.com/IT21182914/Watermarking-OpenCV-Project-UCSC]
   cd watermarking-project
   pip install -r requirements.txt
   python main.py

⭐ Give this project a star if you find it useful!
