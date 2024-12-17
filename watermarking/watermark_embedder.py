import os
import cv2
import numpy as np

def process_images(input_dir, output_dir, bit, alpha):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        watermark = (bit * 2 - 1) * np.random.randn(*img.shape)
        watermarked_img = cv2.addWeighted(img, 1, watermark.astype(np.float32), alpha, 0)
        cv2.imwrite(os.path.join(output_dir, filename), watermarked_img)

def informed_process_images(input_dir, output_dir, bit, alpha):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        watermark = (bit * 2 - 1) * np.random.randn(*img.shape)
        watermark = watermark / np.linalg.norm(watermark)  # Normalize watermark
        watermarked_img = cv2.addWeighted(img, 1, watermark.astype(np.float32), alpha, 0)
        cv2.imwrite(os.path.join(output_dir, filename), watermarked_img)
        print(f"Informed watermark embedded for {filename}")

