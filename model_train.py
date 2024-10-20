from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from data_preparation import prepare_data  # Import dữ liệu đã chuẩn bị
import joblib

# Chuẩn bị dữ liệu
X, y = prepare_data()

# Chia dữ liệu thành tập Train và Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Khởi tạo các mô hình

knn_model_3 = KNeighborsClassifier(n_neighbors=3)  # KNN với k
knn_model_5 = KNeighborsClassifier(n_neighbors=5)
knn_model_7 = KNeighborsClassifier(n_neighbors=7)
knn_model_9 = KNeighborsClassifier(n_neighbors=9)

logistic_model = LogisticRegression(max_iter=1000)  # Logistic Regression

svm_model = SVC(kernel='linear', C=1)  # SVM với kernel tuyến tính


# Huấn luyện KNN
# k=3
print("Huấn luyện KNN...")
knn_model_3.fit(X_train, y_train)
y_pred_knn_3 = knn_model_3.predict(X_test)
print(f"Độ chính xác của KNN: {accuracy_score(y_test, y_pred_knn_3) * 100:.2f}%")
print(classification_report(y_test, y_pred_knn_3))
# k=5
knn_model_5.fit(X_train, y_train)
y_pred_knn_5 = knn_model_5.predict(X_test)
print(f"Độ chính xác của KNN: {accuracy_score(y_test, y_pred_knn_5) * 100:.2f}%")
print(classification_report(y_test, y_pred_knn_5))
# k=7
knn_model_7.fit(X_train, y_train)
y_pred_knn_7 = knn_model_7.predict(X_test)
print(f"Độ chính xác của KNN: {accuracy_score(y_test, y_pred_knn_7) * 100:.2f}%")
print(classification_report(y_test, y_pred_knn_7))
# k=9
knn_model_9.fit(X_train, y_train)
y_pred_knn_9 = knn_model_9.predict(X_test)
print(f"Độ chính xác của KNN: {accuracy_score(y_test, y_pred_knn_9) * 100:.2f}%")
print(classification_report(y_test, y_pred_knn_9))

# Huấn luyện Logistic Regression
print("\nHuấn luyện Logistic Regression...")
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)
print(f"Độ chính xác của Logistic Regression: {accuracy_score(y_test, y_pred_logistic) * 100:.2f}%")
print(classification_report(y_test, y_pred_logistic))

# Huấn luyện SVM
print("\nHuấn luyện SVM...")
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
print(f"Độ chính xác của SVM: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
print(classification_report(y_test, y_pred_svm))

# Lưu các mô hình đã huấn luyện

joblib.dump(knn_model_3, 'knn_model_3.pkl')
joblib.dump(knn_model_5, 'knn_model_5.pkl')
joblib.dump(knn_model_7, 'knn_model_7.pkl')
joblib.dump(knn_model_9, 'knn_model_9.pkl')

joblib.dump(logistic_model, 'logistic_model.pkl')

joblib.dump(svm_model, 'svm_model.pkl')

print("Lưu các mô hình đã huấn luyện thành công!")
