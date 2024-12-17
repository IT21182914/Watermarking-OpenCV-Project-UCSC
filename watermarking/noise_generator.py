import os
import cv2
import numpy as np

def apply_noise(input_dir, output_dir, noise_type):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if noise_type == "gaussian":
            noise = np.random.normal(0, 25, img.shape).astype(np.float32)
            noisy_img = cv2.add(img.astype(np.float32), noise)
        elif noise_type == "shot":
            noisy_img = np.random.poisson(img).astype(np.float32)
        elif noise_type == "salt_pepper":
            noisy_img = img.copy()
            salt = np.random.rand(*img.shape) < 0.02
            pepper = np.random.rand(*img.shape) < 0.02
            noisy_img[salt] = 255
            noisy_img[pepper] = 0
        cv2.imwrite(os.path.join(output_dir, filename), noisy_img)
