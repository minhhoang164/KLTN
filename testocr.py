import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

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
        return image, image

    # Xoay ảnh theo góc trung bình
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return image, rotated_image

# 🔹 Hàm phát hiện vùng chữ và tạo ảnh nhị phân
def detect_text_regions_and_mask_text_only(image):
    """
    Trả về ảnh đã được làm tối phần không có chữ, chỉ giữ lại vùng chữ.
    """
    # Làm nét ảnh
    sharpen_kernel = np.array([[0, -1, 0], 
                               [-1, 5, -1], 
                               [0, -1, 0]])
    sharpened = cv2.filter2D(image, -1, sharpen_kernel)

    # PaddleOCR nhận diện chữ
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi", det_db_box_thresh=0.3)
    result = paddle_ocr.ocr(sharpened, cls=True)

    # Tạo ảnh mask nhị phân
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    for line in result:
        for word in line:
            points = np.array(word[0], dtype=np.int32)
            cv2.fillPoly(mask, [points], 255)

    # Áp mask lên ảnh gốc: tô đen những vùng không có chữ
    text_only_image = cv2.bitwise_and(image, image, mask=mask)

    return text_only_image, mask

# 🔹 Chạy chương trình chính
if __name__ == "__main__":
    image_path = r"E:\HocTap\KLTN\bien_hieu10.png"

    original, rotated = auto_rotate_image(image_path)

    if rotated is not None:
        text_only_image, mask = detect_text_regions_and_mask_text_only(rotated)

        # Hiển thị kết quả
        plt.figure(figsize=(18, 6))

        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        plt.title("Ảnh gốc")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
        plt.title("Ảnh đã xoay")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(text_only_image, cv2.COLOR_BGR2RGB))
        plt.title("Chỉ vùng chữ (phần khác bị đen)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()
