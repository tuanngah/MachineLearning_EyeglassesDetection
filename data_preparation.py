from skimage.feature import hog
import cv2
import os
import numpy as np

def extract_hog_features(image):
    # Trích xuất đặc trưng HOG từ ảnh grayscale
    features, _ = hog(
        image, 
        orientations=9, 
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), 
        block_norm='L2-Hys',
        visualize=True
    )
    return features

def load_data_with_hog(data_dir, label):
    X, y = [], []
    for img_name in os.listdir(data_dir):
        img_path = os.path.join(data_dir, img_name)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Chuyển sang grayscale
        img = cv2.resize(img, (64, 64))  # Resize về 64x64
        features = extract_hog_features(img)  # Trích xuất đặc trưng HOG
        X.append(features)  # Thêm đặc trưng vào danh sách
        y.append(label)  # Gán nhãn
    return X, y

def prepare_data():
    X_celebA_glasses, y_celebA_glasses = load_data_with_hog('dataset/celebA/glasses', 1)
    X_celebA_no_glasses, y_celebA_no_glasses = load_data_with_hog('dataset/celebA/no_glasses', 0)
    X_custom_glasses, y_custom_glasses = load_data_with_hog('dataset/custom/glasses', 1)
    X_custom_no_glasses, y_custom_no_glasses = load_data_with_hog('dataset/custom/no_glasses', 0)

    # Kết hợp tất cả dữ liệu
    X = np.array(X_celebA_glasses + X_celebA_no_glasses + X_custom_glasses + X_custom_no_glasses)
    y = np.array(y_celebA_glasses + y_celebA_no_glasses + y_custom_glasses + y_custom_no_glasses)

    #X = np.array(X_custom_glasses + X_custom_no_glasses)
    #y = np.array(y_custom_glasses + y_custom_no_glasses)

    return X, y

if __name__ == "__main__":
    X, y = prepare_data()
    print(f"Tổng số ảnh: {len(X)}")
    print(f"Số ảnh có kính: {sum(y)}")
    print(f"Số ảnh không kính: {len(y) - sum(y)}")