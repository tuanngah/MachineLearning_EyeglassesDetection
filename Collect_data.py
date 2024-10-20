import cv2
import os

os.makedirs('dataset/custom/glasses', exist_ok=True)
os.makedirs('dataset/custom/no_glasses', exist_ok=True)

# Khởi tạo bộ đếm số lượng ảnh đã lưu
count = {
    'glasses': len(os.listdir('dataset/custom/glasses')),
    'no_glasses': len(os.listdir('dataset/custom/no_glasses'))
}

# Khởi động camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể đọc từ camera.")
        break

    # Hiển thị khung hình từ camera
    cv2.imshow('Capture Data', frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('1'):  # Nhấn '1' nếu đeo kính
        img_name = f'dataset/custom/glasses/img_{count["glasses"]}.jpg'
        cv2.imwrite(img_name, frame)
        print(f"Lưu ảnh có kính: {img_name}")
        count['glasses'] += 1

    elif key == ord('0'):  # Nhấn '0' nếu không đeo kính
        img_name = f'dataset/custom/no_glasses/img_{count["no_glasses"]}.jpg'
        cv2.imwrite(img_name, frame)
        print(f"Lưu ảnh không kính: {img_name}")
        count['no_glasses'] += 1

    elif key == ord('q'):  # Nhấn 'q' để thoát
        print("Thoát chương trình.")
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()


print(f"Số lượng ảnh có kính: {len(os.listdir('dataset/custom/glasses'))}")
print(f"Số lượng ảnh không kính: {len(os.listdir('dataset/custom/no_glasses'))}")
