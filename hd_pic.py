import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

def auto_rotate_image(image_path):
    # Đọc ảnh
    image = cv2.imread(image_path)

    # Kiểm tra ảnh có đọc được không
    if image is None:
        print("❌ Không thể đọc ảnh! Kiểm tra đường dẫn.")
        return None

    # 🔹 Phát hiện chữ bằng PaddleOCR
    paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi", det_db_box_thresh=0.3)
    result = paddle_ocr.ocr(image_path, cls=True)

    # 🔹 Tính toán góc nghiêng trung bình
    angles = []
    for line in result:
        for word in line:
            points = np.array(word[0], dtype=np.float32)
            x1, y1 = points[0]  # Góc trên trái
            x2, y2 = points[1]  # Góc trên phải
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi  # Tính góc xoay
            angles.append(angle)

    # Lấy góc trung bình
    if angles:
        avg_angle = np.mean(angles)
        print(f"📐 Góc xoay trung bình: {avg_angle:.2f}°")
    else:
        print("❌ Không tìm thấy chữ để tính góc xoay.")
        return image

    # 🔹 Xoay ảnh theo góc trung bình
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image

# Đường dẫn ảnh
image_path = r"E:\HocTap\KLTN\bien_hieu5.png"
rotated = auto_rotate_image(image_path)

# Hiển thị ảnh trước & sau khi xoay
if rotated is not None:
    original = cv2.imread(image_path)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ảnh gốc")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title("Ảnh sau khi xoay")

    plt.show()
