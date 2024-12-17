import os
import shutil

def split_dataset(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    set_sizes = [4000, 4000, 5000]
    for i, set_size in enumerate(set_sizes):
        set_dir = os.path.join(output_dir, f"set{i+1}")
        os.makedirs(set_dir, exist_ok=True)
        files = sorted(os.listdir(data_dir))[:set_size]
        for file in files:
            shutil.copy(os.path.join(data_dir, file), os.path.join(set_dir, file))
