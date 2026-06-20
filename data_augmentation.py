import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, affine_transform, binary_dilation, binary_erosion

def shift_image(image, dx, dy):
    shifted = np.zeros_like(image)
    y_start_shift, y_end_shift = max(0, dy), min(16, 16 + dy)
    x_start_shift, x_end_shift = max(0, dx), min(16, 16 + dx)
    y_start_img, y_end_img = max(0, -dy), min(16, 16 - dy)
    x_start_img, x_end_img = max(0, -dx), min(16, 16 - dx)
    shifted[y_start_shift:y_end_shift, x_start_shift:x_end_shift] = image[y_start_img:y_end_img, x_start_img:x_end_img]
    return shifted

def rotate_image(image, max_angle):
    angle = np.random.uniform(-max_angle, max_angle)
    rotated = rotate(image, angle, reshape=False, order=1, mode='constant', cval=0)
    return np.where(rotated > 127, 255, 0).astype(np.uint8)

def shear_image(image, shear_factor):
    transform_matrix = np.array([[1, 0], [shear_factor, 1]])
    center = np.array([8, 8])
    offset = center - np.dot(transform_matrix, center)
    sheared = affine_transform(image, transform_matrix, offset=offset, order=1, mode='constant', cval=0)
    return np.where(sheared > 127, 255, 0).astype(np.uint8)

# [MỚI] Làm đậm nét có định hướng (Chỉ dày thêm 1 pixel, không bị vón cục)
def thicken_stroke_directional(image):
    direction = np.random.randint(0, 4)
    thickened = np.copy(image)
    
    if direction == 0:   # Đẩy sang phải 1 pixel
        thickened[:, 1:] = np.maximum(thickened[:, 1:], image[:, :-1])
    elif direction == 1: # Đẩy xuống dưới 1 pixel
        thickened[1:, :] = np.maximum(thickened[1:, :], image[:-1, :])
    elif direction == 2: # Đẩy sang trái 1 pixel
        thickened[:, :-1] = np.maximum(thickened[:, :-1], image[:, 1:])
    else:                # Đẩy lên trên 1 pixel
        thickened[:-1, :] = np.maximum(thickened[:-1, :], image[1:, :])
        
    return thickened

# [MỚI] Nhiễu viền chữ (Mô phỏng rung tay người vẽ chuột)
def add_edge_jitter(image):
    bin_img = image > 127
    
    # Tìm viền ngoài (Outer edge) và tạo ra các nốt gai nhô ra ngẫu nhiên
    dilated = binary_dilation(bin_img)
    outer_edge = dilated ^ bin_img
    add_mask = np.random.rand(*image.shape) < 0.25 # Xác suất gai mọc
    noisy_img = np.where(outer_edge & add_mask, 255, image)
    
    # Tìm viền trong (Inner edge) và tạo ra các vết mẻ ngẫu nhiên
    eroded = binary_erosion(bin_img)
    inner_edge = bin_img ^ eroded
    remove_mask = np.random.rand(*image.shape) < 0.15 # Xác suất mẻ nét
    noisy_img = np.where(inner_edge & remove_mask, 0, noisy_img)
    
    return noisy_img.astype(np.uint8)


def random_augment_pipeline(image):
    aug_img = np.copy(image)
    
    # 1. Làm đậm nét 1 pixel ngẫu nhiên (Không làm vón cục)
    if np.random.rand() < 0.6:
        aug_img = thicken_stroke_directional(aug_img)

    # 2. Xô lệch (Shear)
    if np.random.rand() < 0.5:
        aug_img = shear_image(aug_img, shear_factor=np.random.uniform(-0.25, 0.25))

    # 3. Xoay (Rotate)
    if np.random.rand() < 0.6:
        aug_img = rotate_image(aug_img, max_angle=12) # Giảm góc xoay xuống 12 độ cho an toàn
        
    # 4. Dịch chuyển (Shift)
    if np.random.rand() < 0.7:
        dx, dy = np.random.randint(-2, 3), np.random.randint(-2, 3)
        aug_img = shift_image(aug_img, dx, dy)
        
    # 5. [MỚI] Thêm nhiễu răng cưa ở viền chữ thay vì nhiễu hạt lung tung
    if np.random.rand() < 0.5:
        aug_img = add_edge_jitter(aug_img)
        
    return aug_img

def generate_training_data(hex_file, samples_per_char=100):
    print("Reading file:", hex_file)
    X_base, y_base = [], []
    
    try:
        with open(hex_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                parts = line.split(':')
                if len(parts) == 2 and len(parts[1]) == 64:
                    unicode_hex = parts[0]
                    bin_str = bin(int(parts[1], 16))[2:].zfill(256)
                    img_array = np.array([int(bit) * 255 for bit in bin_str], dtype=np.uint8).reshape((16, 16))
                    
                    X_base.append(img_array)
                    y_base.append(int(unicode_hex, 16))
    except FileNotFoundError:
        print("File not found")
        return None, None
        
    print("Loaded characters:", len(X_base))
    print("Making more data...")

    X_train, y_train = [], []

    for i in range(len(X_base)):
        base_img, label = X_base[i], y_base[i]
        
        X_train.append(base_img)
        y_train.append(label)
        
        for _ in range(samples_per_char - 1):
            X_train.append(random_augment_pipeline(base_img))
            y_train.append(label)

    X_train, y_train = np.array(X_train), np.array(y_train)

    print("Shuffling data...")
    indices = np.arange(len(X_train))
    np.random.shuffle(indices)
    
    return X_train[indices], y_train[indices]

def preview_dataset(X, y, num_samples=10):
    print("Showing images...")
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Preview Dataset (Edge Noise & Directional Thicken)")
    
    axes = axes.flatten()
    for i in range(min(num_samples, len(X))):
        axes[i].imshow(X[i], cmap='gray')
        axes[i].set_title("Label: " + chr(y[i]))
        axes[i].axis('off')
        
    plt.tight_layout()
    plt.show()

def save_dataset(X, y, output_file):
    print("Saving file...")
    np.savez_compressed(output_file, X=X, y=y)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print("Saved:", output_file)
    print("Size:", round(file_size_mb, 2), "MB")

def main():
    INPUT_FILE = "keyboard_dataset_16x16.hex"
    OUTPUT_FILE = "unifont_train_data_15.npz"
    SAMPLES_PER_CHAR = 100
    
    X_train, y_train = generate_training_data(INPUT_FILE, SAMPLES_PER_CHAR)
    
    if X_train is not None:
        preview_dataset(X_train, y_train, num_samples=10)
        save_dataset(X_train, y_train, OUTPUT_FILE)

if __name__ == "__main__":
    main()