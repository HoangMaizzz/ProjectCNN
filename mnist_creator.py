import numpy as np
import cv2
from sklearn.datasets import fetch_openml

def main():
    print("⏳ Đang tải dữ liệu EMNIST Balanced (Mã 41039)...")
    emnist = fetch_openml(data_id=41039, cache=True, as_frame=False, parser='auto')
    
    X_raw = emnist.data
    y_raw = emnist.target.astype(int)
    
    # Số lượng ảnh bạn muốn lấy cho MỖI KÝ TỰ (Thay đổi tùy ý)
    # 1000 ảnh x 36 ký tự = 36.000 ảnh tổng cộng. Rất cân bằng và học đủ nhanh!
    SAMPLES_PER_CLASS = 1000 
    
    X_balanced = []
    y_balanced = []
    
    print("⚖️ Đang chia đều dữ liệu cho 36 ký tự (0-9 và A-Z)...")
    # Duyệt qua 36 nhãn (0 đến 9 là Số, 10 đến 35 là Chữ A-Z)
    for class_id in range(36):
        # Tìm tất cả vị trí của ký tự hiện tại
        indices = np.where(y_raw == class_id)[0]
        
        # Chỉ bốc đúng số lượng quy định (1000 ảnh) để cân bằng
        selected_indices = indices[:SAMPLES_PER_CLASS]
        
        X_balanced.append(X_raw[selected_indices])
        y_balanced.append(y_raw[selected_indices])
        
    # Gom tất cả lại thành 1 mảng lớn
    X_raw = np.vstack(X_balanced)
    y_raw = np.concatenate(y_balanced)
    
    num_samples = len(X_raw)
    X_resized = np.zeros((num_samples, 16, 16), dtype=np.uint8)
    y_unicode = np.zeros(num_samples, dtype=int)
    
    print(f"Tổng số ảnh: {num_samples}. Đang ép về 16x16...")
    for i in range(num_samples):
        # Lật và xoay ảnh đúng chiều
        img_28 = X_raw[i].reshape((28, 28)).T.astype(np.uint8)
        
        # Bóp nhỏ
        img_16 = cv2.resize(img_28, (16, 16), interpolation=cv2.INTER_AREA)
        _, img_bin = cv2.threshold(img_16, 127, 255, cv2.THRESH_BINARY)
        X_resized[i] = img_bin
        
        # Xử lý Nhãn chuẩn Unicode
        if y_raw[i] <= 9:
            # Nếu là số từ 0-9: Cộng 48 để khớp chuẩn (Ví dụ: số 0 -> 48)
            y_unicode[i] = y_raw[i] + 48
        else:
            # Nếu là chữ từ A-Z (Nhãn 10-35): Cộng 55 để khớp chuẩn (Ví dụ: 10 -> 65 tức là 'A')
            y_unicode[i] = y_raw[i] + 55
            
    print("Đang xáo trộn (Shuffle) để AI không học vẹt...")
    shuffle_idx = np.arange(num_samples)
    np.random.shuffle(shuffle_idx)
    X_resized = X_resized[shuffle_idx]
    y_unicode = y_unicode[shuffle_idx]
            
    print("💾 Đang lưu file...")
    np.savez_compressed("emnist_perfect_16x16.npz", X=X_resized, y=y_unicode)
    print("🎉 XONG! File 'emnist_perfect_16x16.npz' đã ra lò, chuẩn không cần chỉnh!")

if __name__ == "__main__":
    main()