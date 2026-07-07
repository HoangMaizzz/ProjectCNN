import numpy as np
import cv2
from sklearn.datasets import fetch_openml

def main():
    print("⏳ Đang tải dữ liệu EMNIST Balanced (Mã 41039 - 47 Ký tự)...")
    emnist = fetch_openml(data_id=41039, cache=True, as_frame=False, parser='auto')
    
    X_raw = emnist.data
    y_raw = emnist.target.astype(int)
    
    # Lấy 1000 ảnh cho mỗi ký tự -> Tổng 47.000 ảnh
    SAMPLES_PER_CLASS = 1000 
    
    X_balanced = []
    y_balanced = []
    
    print("⚖️ Đang bốc đều 1000 ảnh cho mỗi ký tự (Tổng 47.000 ảnh)...")
    for class_id in range(47):
        indices = np.where(y_raw == class_id)[0]
        selected_indices = indices[:SAMPLES_PER_CLASS]
        
        X_balanced.append(X_raw[selected_indices])
        y_balanced.append(y_raw[selected_indices])
        
    X_raw = np.vstack(X_balanced)
    y_raw = np.concatenate(y_balanced)
    
    num_samples = len(X_raw)
    X_resized = np.zeros((num_samples, 16, 16), dtype=np.uint8)
    y_unicode = np.zeros(num_samples, dtype=int)
    
    print(f"Đang ép {num_samples} ảnh về 16x16 và xử lý nhãn thông minh...")
    
    # Từ điển dịch 11 chữ in thường đặc thù sang mã chuẩn máy tính
    lowercase_map = {
        36: 97,  # a
        37: 98,  # b
        38: 100, # d
        39: 101, # e
        40: 102, # f
        41: 103, # g
        42: 104, # h
        43: 110, # n
        44: 113, # q
        45: 114, # r
        46: 116  # t
    }
    
    for i in range(num_samples):
        # Lật và xoay ảnh đúng chiều
        img_28 = X_raw[i].reshape((28, 28)).T.astype(np.uint8)
        
        # Bóp nhỏ
        img_16 = cv2.resize(img_28, (16, 16), interpolation=cv2.INTER_AREA)
        _, img_bin = cv2.threshold(img_16, 127, 255, cv2.THRESH_BINARY)
        X_resized[i] = img_bin
        
        # --- BỘ LỌC NHÃN 47 CLASSES CHUẨN XÁC ---
        if y_raw[i] <= 9:
            # Nhóm 1: Số 0-9
            y_unicode[i] = y_raw[i] + 48
        elif y_raw[i] <= 35:
            # Nhóm 2: Chữ in Hoa (và các chữ in thường giống in hoa bị gộp vào)
            y_unicode[i] = y_raw[i] + 55
        else:
            # Nhóm 3: 11 chữ in thường đặc biệt
            y_unicode[i] = lowercase_map[y_raw[i]]
            
    print("Đang xáo trộn (Shuffle) để AI không học vẹt...")
    shuffle_idx = np.arange(num_samples)
    np.random.shuffle(shuffle_idx)
    X_resized = X_resized[shuffle_idx]
    y_unicode = y_unicode[shuffle_idx]
            
    print("💾 Đang lưu file...")
    np.savez_compressed("emnist_47_classes_16x16.npz", X=X_resized, y=y_unicode)
    print("🎉 XONG! File 'emnist_47_classes_16x16.npz' đã sẵn sàng để quăng lên Kaggle!")

if __name__ == "__main__":
    main()