import cv2
import numpy as np

def rotate_to_horizontal(image_path, save_path="bien_hieu_rotated.png"):
    # Đọc ảnh
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Lỗi: Không thể đọc ảnh. Kiểm tra lại đường dẫn!")
        return

    # Chuyển ảnh sang grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Làm mờ để giảm nhiễu
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Phát hiện cạnh bằng Canny
    edges = cv2.Canny(blurred, 50, 150)

    # Tìm contour lớn nhất (giả định là biển hiệu)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Không tìm thấy biển hiệu!")
        return

    largest_contour = max(contours, key=cv2.contourArea)

    # Xác định hình chữ nhật bao quanh biển hiệu
    rect = cv2.minAreaRect(largest_contour)
    angle = rect[-1]  # Góc xoay ban đầu

    # Chiều dài & chiều rộng của biển hiệu
    width, height = rect[1]

    # Tính hai góc so với phương ngang
    angle1 = abs(angle)  # Góc theo cạnh ngắn
    angle2 = abs(90 + angle)  # Góc theo cạnh dài

    # Chọn góc nhỏ hơn để xoay
    rotate_angle = -angle1 if angle1 < angle2 else -angle2

    # Xoay ảnh
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, rotate_angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # Lưu ảnh đã xoay
    cv2.imwrite(save_path, rotated)
    print(f"✅ Ảnh đã xoay và lưu tại: {save_path}")

# Chạy thử
rotate_to_horizontal("bien_hieu5.png")
