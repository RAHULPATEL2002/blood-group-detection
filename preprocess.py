import os
import cv2

def preprocess_images(input_dir, image_size=(128, 128)):
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if os.path.isdir(folder_path):
            for file in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file)
                if file_path.endswith(('.jpg', '.png')):
                    try:
                        img = cv2.imread(file_path)
                        if img is None:
                            print(f"Warning: {file_path} is not a valid image file.")
                            continue
                        img = cv2.resize(img, image_size)
                        cv2.imwrite(file_path, img)  # Overwrite original image
                    except Exception as e:
                        print(f"Error processing {file_path}: {e}")

if __name__ == "__main__":
    preprocess_images('data/train')
    preprocess_images('data/validation')
    preprocess_images('data/test')
    print("✅ Image preprocessing completed successfully!")
