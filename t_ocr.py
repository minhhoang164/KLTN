import cv2
import numpy as np
import matplotlib.pyplot as plt
from paddleocr import PaddleOCR
import pytesseract

# Cấu hình đường dẫn Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Đọc ảnh
image_path = "bien_hieu3.png"
image = cv2.imread(image_path)

# Khởi tạo PaddleOCR
paddle_ocr = PaddleOCR(use_angle_cls=True, lang="vi", det_db_box_thresh=0.3)
result = paddle_ocr.ocr(image_path, cls=True)

# Lọc ra những vùng có tọa độ hợp lệ
valid_boxes = []
for line in result:
    for word in line:
        if isinstance(word[0], list):  # Kiểm tra xem có tọa độ không
            valid_boxes.append(word)

# Sắp xếp theo tọa độ y (từ trên xuống)
valid_boxes.sort(key=lambda x: np.min(np.array(x[0]), axis=0)[1])

# Chứa kết quả nhận diện
text_tesseract = ""

# Xử lý từng vùng chữ
for word in valid_boxes:
    points = np.array(word[0], dtype=np.int32)
    cv2.polylines(image, [points], isClosed=True, color=(0, 255, 0), thickness=2)

    # Xác định tọa độ vùng chữ
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    roi = image[y_min:y_max, x_min:x_max]

    # Tiền xử lý ảnh để Tesseract nhận diện tốt hơn
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Chuyển sang grayscale
    roi_thresh = cv2.threshold(roi_gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # OCR với Tesseract
    text = pytesseract.image_to_string(roi_thresh, lang="vie").strip()
    if text:
        text_tesseract += text + " "

# Nếu không nhận diện được chữ, hiển thị thông báo
if not text_tesseract.strip():
    text_tesseract = "Không nhận diện được chữ."

# Hiển thị ảnh bên trái, text bên phải
plt.figure(figsize=(12, 6))

# Ảnh với vùng chữ khoanh lại
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")

# Text nhận diện từ Tesseract
plt.subplot(1, 2, 2)
plt.text(0, 0.5, text_tesseract.strip(), fontsize=14, color="black", wrap=True, verticalalignment='center')
plt.axis("off")

plt.show()
