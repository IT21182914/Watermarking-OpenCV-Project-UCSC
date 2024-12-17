import os
import cv2
import numpy as np

def generate_gaussian_pattern(shape):
    return np.random.randn(*shape)

def process_detection(input_dir, Wr):
    results = {'bit_0': 0, 'bit_1': 0, 'no_watermark': 0, 'correlations': []}
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        correlation = np.sum(img * Wr)
        results['correlations'].append(correlation)
        if correlation > 50:
            results['bit_1'] += 1
        elif correlation < -50:
            results['bit_0'] += 1
        else:
            results['no_watermark'] += 1
    return results
