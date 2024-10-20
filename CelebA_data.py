import os
import shutil
import pandas as pd

# Đường dẫn đến thư mục ảnh và tệp nhãn của CelebA
celebA_path = r'D:\K16AIRB\Năm học\2024-2025\Kì 1\Học máy\Project_Glasses or not\celeba\img_align_celeba'
attr_file = r'D:\K16AIRB\Năm học\2024-2025\Kì 1\Học máy\Project_Glasses or not\celeba\list_attr_celeba.csv'

# Tạo các thư mục để lưu ảnh đã lọc
os.makedirs('dataset/celebA/glasses', exist_ok=True)
os.makedirs('dataset/celebA/no_glasses', exist_ok=True)

def filter_celeba_data():
    # Đọc tệp thuộc tính bằng Pandas
    attrs = pd.read_csv(attr_file)

    # Lọc và sao chép ảnh vào thư mục tương ứng
    for index, row in attrs.iterrows():
        img_name = row['image_id']
        glasses_label = row['Eyeglasses']

        # Đường dẫn đến ảnh nguồn
        src_path = os.path.join(celebA_path, img_name)

        # Kiểm tra nhãn và xác định thư mục đích
        if glasses_label == 1:  # Có kính
            dest_path = f'dataset/celebA/glasses/{img_name}'
        else:  # Không kính
            dest_path = f'dataset/celebA/no_glasses/{img_name}'

        # Sao chép ảnh vào thư mục đích
        if os.path.exists(src_path):
            shutil.copy(src_path, dest_path)

    print("Lọc và sao chép ảnh hoàn tất!")

if __name__ == "__main__":
    filter_celeba_data()

    print(f'Số lượng ảnh có kính: {len(os.listdir("dataset/celebA/glasses"))}')
    print(f'Số lượng ảnh không kính: {len(os.listdir("dataset/celebA/no_glasses"))}')
