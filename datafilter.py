import os

def pad_to_16x16_hex(pixel_hex):
    """
    Biến đổi chuỗi hex 8x16 (32 ký tự) thành 16x16 (64 ký tự) bằng cách căn giữa.
    Nếu đã là 64 ký tự thì giữ nguyên.
    """
    if len(pixel_hex) == 64:
        return pixel_hex
    elif len(pixel_hex) == 32:
        padded_hex = ""
        # Duyệt qua 16 hàng, mỗi hàng lấy 2 ký tự hex (8 pixel)
        for i in range(16):
            row_hex = pixel_hex[i*2 : i*2+2]
            # Kẹp thêm số '0' (4 bit trống) vào bên trái và bên phải
            padded_hex += "0" + row_hex + "0"
        return padded_hex
    else:
        return None

def extract_and_format_keyboard_chars(input_file, output_file):
    keyboard_chars = list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789"
        "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
    )
    
    target_hex_codes = {f"{ord(c):04X}" for c in keyboard_chars}
    
    print(f"🔍 Đang tìm và chuẩn hóa {len(target_hex_codes)} ký tự bàn phím...")
    
    extracted_lines = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f_in:
            for line in f_in:
                line = line.strip()
                if not line: 
                    continue
                
                parts = line.split(':')
                if len(parts) == 2:
                    unicode_hex = parts[0]
                    pixel_hex = parts[1]
                    
                    if unicode_hex in target_hex_codes:
                        # Thực hiện bước Đệm (Padding) ép về chuẩn 16x16
                        formatted_pixel_hex = pad_to_16x16_hex(pixel_hex)
                        
                        if formatted_pixel_hex:
                            # Tạo dòng mới với format chuẩn chỉnh
                            extracted_lines.append(f"{unicode_hex}:{formatted_pixel_hex}")
                            
    except FileNotFoundError:
        print(f"❌ Không tìm thấy file gốc '{input_file}'.")
        return
        
    if extracted_lines:
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for line in extracted_lines:
                f_out.write(line + '\n')
                
        print(f"🎉 Xuất sắc! Đã trích xuất và ÉP CHUẨN 16x16 thành công {len(extracted_lines)} ký tự.")
        print(f"💾 File dataset mini đã sẵn sàng tại: {output_file}")
    else:
        print("⚠️ Không tìm thấy ký tự nào khớp trong file gốc.")

# --- CHẠY THỬ ---
file_goc = "dataset_chuan.hex" 
file_mini = "keyboard_dataset_16x16.hex" # Đổi tên file để bạn dễ nhận biết

extract_and_format_keyboard_chars(file_goc, file_mini)