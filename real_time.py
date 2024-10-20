import cv2
import joblib
from skimage.feature import hog

# Tải mô hình đã huấn luyện (chọn mô hình bạn muốn triển khai)
#model = joblib.load('logistic_model.pkl')
#model = joblib.load('knn_model_3.pkl')
#model = joblib.load('knn_model_5.pkl')
#model = joblib.load('knn_model_7.pkl')
#model = joblib.load('knn_model_9.pkl')
model = joblib.load('svm_model.pkl')  

# Hàm trích xuất đặc trưng HOG từ ảnh
def extract_hog_features(image):
    features, _ = hog(
        image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        visualize=True
    )
    return features

# Hàm để nhận diện trong thời gian thực từ camera
def real_time_detection():
    cap = cv2.VideoCapture(0)  # Mở camera

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Không thể đọc từ camera.")
            break

        # Chuyển sang grayscale và resize về 64x64
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img_resized = cv2.resize(gray, (64, 64))

        # Trích xuất đặc trưng HOG từ ảnh
        features = extract_hog_features(img_resized).reshape(1, -1)

        # Dự đoán bằng mô hình đã chọn
        prediction = model.predict(features)[0]
        label = "With Glasses" if prediction == 1 else "No Glasses"

        # Hiển thị kết quả trên khung hình
        cv2.putText(frame, label, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Real-Time Detection', frame)

        # Nhấn 'ESC' để thoát
        if cv2.waitKey(1) & 0xFF == 27:
            print("Thoát chương trình.")
            break

    cap.release()  # Giải phóng camera
    cv2.destroyAllWindows()  # Đóng cửa sổ

if __name__ == "__main__":
    real_time_detection()
