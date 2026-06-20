import os

def reconstruct_hex_file(messy_folder_path, output_file_name):
    print(f"🕵️ Đang quét toàn bộ file trong: {messy_folder_path}...")
    
    if not os.path.isdir(messy_folder_path):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{messy_folder_path}'.")
        return

    file_names = os.listdir(messy_folder_path)
    recovered_data = []

    for file_name in file_names:
        if "_" not in file_name:
            continue
            
        parts = file_name.split("_")
        if len(parts) >= 2:
            unicode_hex = parts[0]
            # Bỏ đi phần đuôi mở rộng (như .txt) nếu Windows tự thêm vào
            pixel_hex = parts[1].split('.')[0] 
            
            try:
                # Ép kiểu unicode_hex sang số nguyên để lát nữa sắp xếp cho chuẩn
                sort_key = int(unicode_hex, 16)
                recovered_data.append((sort_key, unicode_hex, pixel_hex))
            except ValueError:
                # Bỏ qua nếu tên file bị lỗi (không phải là số Hex)
                continue

    if not recovered_data:
        print("❌ Không tìm thấy dữ liệu hợp lệ nào để khôi phục.")
        return

    print(f"✅ Đã thu gom được {len(recovered_data)} dòng dữ liệu.")
    print("⏳ Đang sắp xếp lại theo thứ tự từ điển Unicode...")
    
    # Sắp xếp lại danh sách từ bé đến lớn theo mã Unicode (giống hệt file gốc)
    recovered_data.sort(key=lambda x: x[0])

    print(f"💾 Đang ghi dữ liệu vào file mới: {output_file_name}...")
    
    # Mở file mới để ghi dữ liệu vào
    with open(output_file_name, 'w', encoding='utf-8') as f:
        for item in recovered_data:
            # item[1] là Unicode, item[2] là chuỗi pixel
            f.write(f"{item[1]}:{item[2]}\n")

    print("🎉 HOÀN TẤT! File của bạn đã được khôi phục nguyên trạng.")
    print("💡 Bây giờ bạn có thể xóa cái thư mục chứa 57.000 file kia đi cho nhẹ máy!")

# --- CHẠY THỬ ---
# 1. Tên thư mục chứa đống file rỗng hiện tại của bạn
messy_folder = "unifont-17.0.04.hex" 

# 2. Tên file chuẩn, sạch sẽ mà bạn muốn tạo ra
clean_file = "dataset_chuan.hex"     

reconstruct_hex_file(messy_folder, clean_file)