import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR

# Đọc ảnh
image_path = "bien_hieu3.png"
image = cv2.imread(image_path)

# Khởi tạo PaddleOCR
paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi", det_db_box_thresh=0.3)  # Giảm threshold nếu cần
result = paddle_ocr.ocr(image_path, cls=True)

# Chứa kết quả từ OCR
text_paddleocr = ""

# Xử lý từng vùng chữ
if result is not None:
    for line in result:
        for word in line:
            points = np.array(word[0], dtype=np.int32)
            cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

            # Lưu text nhận diện được
            text_paddleocr += word[1][0] + " "

# Nếu không có chữ, hiển thị thông báo
if not text_paddleocr.strip():
    text_paddleocr = "Không nhận diện được chữ."

# Hiển thị ảnh bên trái và text bên phải
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1, 2, 2)
plt.text(0, 0.5, text_paddleocr.strip(), fontsize=14, color="black", wrap=True, verticalalignment='center')
plt.axis("off")

plt.show()
