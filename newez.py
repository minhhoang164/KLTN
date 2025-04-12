import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import easyocr

# 🔹 Hàm tự động xoay ảnh
def auto_rotate_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Không thể đọc ảnh! Kiểm tra đường dẫn.")
        return None, None
    
    # Phát hiện chữ bằng PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi", det_db_box_thresh=0.5)
    result = paddle_ocr.ocr(image_path, cls=True)

    # Tính toán góc xoay trung bình
    angles = []
    for line in result:
        for word in line:
            points = np.array(word[0], dtype=np.float32)
            x1, y1 = points[0]  # Góc trên trái
            x2, y2 = points[1]  # Góc trên phải
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            angles.append(angle)

    if angles:
        avg_angle = np.mean(angles)
        print(f"📐 Góc xoay trung bình: {avg_angle:.2f}°")
    else:
        print("❌ Không tìm thấy chữ để tính góc xoay.")
        return image, image  # Trả về ảnh gốc nếu không xoay

    # Xoay ảnh theo góc trung bình
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return image, rotated_image  # Trả về ảnh gốc và ảnh đã xoay

# 🔹 Hàm OCR
def ocr_after_rotation(original_image, rotated_image, ground_truth):
    if rotated_image is None:
        print("❌ Ảnh bị lỗi, không thể OCR.")
        return

    # Làm nét ảnh
    sharpen_kernel = np.array([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]])
    sharpened = cv2.filter2D(rotated_image, -1, sharpen_kernel)

    # OCR với PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi", det_db_box_thresh=0.3)
    result = paddle_ocr.ocr(sharpened, cls=True)

    # OCR với EasyOCR
    easy_ocr = easyocr.Reader(["vi", "en"])
    text_easyocr = ""

    # Xử lý từng vùng chữ
    for line in result:
        for word in line:
            points = np.array(word[0], dtype=np.int32)
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            roi = sharpened[y_min:y_max, x_min:x_max]
            
            # Phóng to vùng chữ
            roi_resized = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            
            # Làm nét lại sau khi phóng to
            # roi_sharpened = cv2.filter2D(roi_resized, -1, sharpen_kernel)
            
            # Nhận diện chữ bằng EasyOCR
            easy_result = easy_ocr.readtext(roi_resized)
            for easy_word in easy_result:
                text_easyocr += easy_word[1] + " "

    # Tính độ chính xác
    recognized_text = text_easyocr.strip()
    correct_chars = sum(1 for a, b in zip(recognized_text, ground_truth) if a == b)
    accuracy = correct_chars / max(len(ground_truth), 1) * 100
    print(f"✅ Độ chính xác OCR: {accuracy:.2f}%")

    # Hiển thị ảnh gốc & kết quả OCR
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ảnh gốc + vùng chữ")

    plt.subplot(1, 2, 2)
    plt.text(0, 0.5, recognized_text, fontsize=14, color="black", wrap=True, verticalalignment='center')
    plt.axis("off")
    plt.title("Kết quả OCR")

    plt.show()

# 🔹 Chạy chương trình
image_path = r"E:\HocTap\KLTN\bien_hieu3.png"
ground_truth = "Nhập văn bản gốc tại đây"  # Cập nhật văn bản gốc để so sánh
original, rotated = auto_rotate_image(image_path)
ocr_after_rotation(original, rotated, ground_truth)
