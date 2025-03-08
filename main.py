import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from watermarking.dataset_splitter import split_dataset
from watermarking.noise_generator import apply_noise
from watermarking.watermark_detector import generate_gaussian_pattern, process_detection
from watermarking.watermark_embedder import process_images, informed_process_images

def check_directory_exists(directory):
    return os.path.exists(directory) and len(os.listdir(directory)) > 0

def plot_correlation_values(correlations, title, threshold_upper, threshold_lower):
    plt.figure(figsize=(10, 6))
    plt.hist(correlations, bins=50, color='skyblue')
    plt.axvline(threshold_upper, color='red', linestyle='--', label='Upper Threshold')
    plt.axvline(threshold_lower, color='green', linestyle='--', label='Lower Threshold')
    plt.title(title)
    plt.xlabel("Correlation Value")
    plt.ylabel("Number of Images")
    plt.legend()
    plt.show()

def calculate_false_positive_rate(correlations, threshold_upper, threshold_lower):
    false_positives = sum((c < threshold_upper and c > threshold_lower) for c in correlations)
    return false_positives / len(correlations)

if __name__ == "__main__":
    # Activity 1: Split Dataset
    if not check_directory_exists("output/set1"):
        print("Splitting dataset...")
        split_dataset("data/lfw", "output")
    else:
        print("Dataset already split. Skipping...")

    # Activity 2: Embed Blind Watermarks ,
    if not check_directory_exists("output/watermarked_set1"):
        print("Embedding blind watermarks...")
        process_images("output/set1", "output/watermarked_set1", bit=0, alpha=1.0)
        process_images("output/set2", "output/watermarked_set2", bit=1, alpha=1.0)
    else:
        print("Blind watermarks already embedded. Skipping...")

    # Activity 4: Add Noise
    if not check_directory_exists("output/gaussian_noise_set1"):
        print("Adding Gaussian noise to Set1...")
        apply_noise("output/watermarked_set1", "output/gaussian_noise_set1", noise_type="gaussian")
    if not check_directory_exists("output/shot_noise_set1"):
        print("Adding Shot noise to Set1...")
        apply_noise("output/watermarked_set1", "output/shot_noise_set1", noise_type="shot")
    if not check_directory_exists("output/salt_pepper_noise_set1"):
        print("Adding Salt & Pepper noise to Set1...")
        apply_noise("output/watermarked_set1", "output/salt_pepper_noise_set1", noise_type="salt_pepper")

    if not check_directory_exists("output/gaussian_noise_set2"):
        print("Adding Gaussian noise to Set2...")
        apply_noise("output/watermarked_set2", "output/gaussian_noise_set2", noise_type="gaussian")
    if not check_directory_exists("output/shot_noise_set2"):
        print("Adding Shot noise to Set2...")
        apply_noise("output/watermarked_set2", "output/shot_noise_set2", noise_type="shot")
    if not check_directory_exists("output/salt_pepper_noise_set2"):
        print("Adding Salt & Pepper noise to Set2...")
        apply_noise("output/watermarked_set2", "output/salt_pepper_noise_set2", noise_type="salt_pepper")

    # Generate Gaussian Pattern
    example_image_path = "output/watermarked_set1/James_May_0001.jpg"
    Wr = generate_gaussian_pattern(cv2.imread(example_image_path, cv2.IMREAD_GRAYSCALE).shape)

    # Activity 5: Detect Watermarks and Plot Results for Set1
    print("Detecting watermarks for Set1...")
    results_set1 = process_detection("output/watermarked_set1", Wr)
    results_gaussian = process_detection("output/gaussian_noise_set1", Wr)
    results_shot = process_detection("output/shot_noise_set1", Wr)
    results_salt_pepper = process_detection("output/salt_pepper_noise_set1", Wr)

    print("Results for Watermarked Set 1:")
    print(results_set1)
    print("Results for Gaussian Noise Set (Set1):")
    print(results_gaussian)
    print("Results for Shot Noise Set (Set1):")
    print(results_shot)
    print("Results for Salt & Pepper Noise Set (Set1):")
    print(results_salt_pepper)

    plot_correlation_values(results_set1['correlations'], "Watermarked Set 1", 100, -50)
    plot_correlation_values(results_gaussian['correlations'], "Gaussian Noise Set (Set1)", 100, -50)
    plot_correlation_values(results_shot['correlations'], "Shot Noise Set (Set1)", 100, -50)
    plot_correlation_values(results_salt_pepper['correlations'], "Salt & Pepper Noise Set (Set1)", 100, -50)

    print("Calculating False Positive Rates for Set1...")
    fpr_gaussian = calculate_false_positive_rate(results_gaussian['correlations'], 100, -50)
    fpr_shot = calculate_false_positive_rate(results_shot['correlations'], 100, -50)
    fpr_salt_pepper = calculate_false_positive_rate(results_salt_pepper['correlations'], 100, -50)

    print(f"False Positive Rate for Gaussian Noise (Set1): {fpr_gaussian}")
    print(f"False Positive Rate for Shot Noise (Set1): {fpr_shot}")
    print(f"False Positive Rate for Salt & Pepper Noise (Set1): {fpr_salt_pepper}")

    # Activity 5: Detect Watermarks and Plot Results for Set2
    print("Detecting watermarks for Set2...")
    results_set2 = process_detection("output/watermarked_set2", Wr)
    results_gaussian_set2 = process_detection("output/gaussian_noise_set2", Wr)
    results_shot_set2 = process_detection("output/shot_noise_set2", Wr)
    results_salt_pepper_set2 = process_detection("output/salt_pepper_noise_set2", Wr)

    print("Results for Watermarked Set 2:")
    print(results_set2)
    print("Results for Gaussian Noise Set (Set2):")
    print(results_gaussian_set2)
    print("Results for Shot Noise Set (Set2):")
    print(results_shot_set2)
    print("Results for Salt & Pepper Noise Set (Set2):")
    print(results_salt_pepper_set2)

    plot_correlation_values(results_set2['correlations'], "Watermarked Set 2", 100, -50)
    plot_correlation_values(results_gaussian_set2['correlations'], "Gaussian Noise Set (Set2)", 100, -50)
    plot_correlation_values(results_shot_set2['correlations'], "Shot Noise Set (Set2)", 100, -50)
    plot_correlation_values(results_salt_pepper_set2['correlations'], "Salt & Pepper Noise Set (Set2)", 100, -50)

    print("Calculating False Positive Rates for Set2...")
    fpr_gaussian_set2 = calculate_false_positive_rate(results_gaussian_set2['correlations'], 100, -50)
    fpr_shot_set2 = calculate_false_positive_rate(results_shot_set2['correlations'], 100, -50)
    fpr_salt_pepper_set2 = calculate_false_positive_rate(results_salt_pepper_set2['correlations'], 100, -50)

    print(f"False Positive Rate for Gaussian Noise (Set2): {fpr_gaussian_set2}")
    print(f"False Positive Rate for Shot Noise (Set2): {fpr_shot_set2}")
    print(f"False Positive Rate for Salt & Pepper Noise (Set2): {fpr_salt_pepper_set2}")

    # Activity 6: Informed Watermark Embedding
    if not check_directory_exists("output/informed_watermarked_set1"):
        print("Embedding watermarks (Informed Embedder with alpha=0.4)...")
        informed_process_images("output/set1", "output/informed_watermarked_set1", bit=0, alpha=0.4)
    else:
        print("Informed watermarks already embedded. Skipping...")

    print("ALL activities completed successfully!")
