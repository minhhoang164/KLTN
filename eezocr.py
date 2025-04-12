import cv2
import numpy as np
import matplotlib.pyplot as plt
import easyocr

# 🔹 Hàm tự động xoay ảnh
def auto_rotate_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể đọc ảnh! Kiểm tra đường dẫn.")
        return None, None
    
    # Chuyển ảnh về grayscale
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Làm nét ảnh
    sharpen_kernel = np.array([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)
    
    return image, sharpened  # Trả về ảnh gốc và ảnh đã xử lý

# 🔹 Hàm OCR
def ocr_after_rotation(original_image, processed_image):
    if processed_image is None:
        print("❌ Ảnh bị lỗi, không thể OCR.")
        return

    # OCR với EasyOCR
    easy_ocr = easyocr.Reader(["vi", "en"])
    result = easy_ocr.readtext(processed_image)
    text_easyocr = " ".join([word[1] for word in result])

    # Hiển thị ảnh gốc & kết quả OCR
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ảnh gốc")

    plt.subplot(1, 2, 2)
    plt.text(0, 0.5, text_easyocr.strip(), fontsize=14, color="black", wrap=True, verticalalignment='center')
    plt.axis("off")
    plt.title("Kết quả OCR")

    plt.show()

# 🔹 Chạy chương trình
image_path = r"E:\HocTap\KLTN\bien_hieu9.png"
original, processed = auto_rotate_image(image_path)
ocr_after_rotation(original, processed)