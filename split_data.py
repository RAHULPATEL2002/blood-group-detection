import os
import shutil
import random
from sklearn.model_selection import train_test_split

# Input and output directories
input_dir = r"D:\pythonimageprocessing\dataset_folder"
output_dir = r"D:\pythonimageprocessing\data"

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create main output folders
for folder in ["train", "validation", "test"]:
    os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

# Get all top-level blood group class folders
class_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]

for class_folder in class_folders:
    class_path = os.path.join(input_dir, class_folder)

    # Get subclass folders (e.g., high, normal, low)
    subclass_folders = [
        f for f in os.listdir(class_path)
        if os.path.isdir(os.path.join(class_path, f))
    ]

    for subclass_folder in subclass_folders:
        subclass_path = os.path.join(class_path, subclass_folder)

        # Get all image paths in the subclass
        image_paths = [
            os.path.join(subclass_path, file)
            for file in os.listdir(subclass_path)
            if file.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]

        # Shuffle and split
        random.shuffle(image_paths)
        train_imgs, temp_imgs = train_test_split(image_paths, test_size=(1 - train_ratio), random_state=42)
        val_imgs, test_imgs = train_test_split(temp_imgs, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=42)

        # Copy images to respective output folders, preserving class + subclass structure
        for img_list, split in zip([train_imgs, val_imgs, test_imgs], ["train", "validation", "test"]):
            target_dir = os.path.join(output_dir, split, class_folder, subclass_folder)
            os.makedirs(target_dir, exist_ok=True)

            for img_path in img_list:
                filename = os.path.basename(img_path)
                dst_path = os.path.join(target_dir, filename)

                # Avoid filename collision
                if os.path.exists(dst_path):
                    base, ext = os.path.splitext(filename)
                    count = 1
                    while os.path.exists(dst_path):
                        dst_path = os.path.join(target_dir, f"{base}_{count}{ext}")
                        count += 1

                shutil.copy(img_path, dst_path)

        print(f"✅ Split {class_folder}/{subclass_folder}: "
              f"{len(train_imgs)} train, {len(val_imgs)} validation, {len(test_imgs)} test")

print("🎯 Dataset split completed successfully!")
