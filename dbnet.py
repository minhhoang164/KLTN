import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms


# ===== DBNet đơn giản (Text Detection) =====
class DBNet(nn.Module):
    def __init__(self):
        super(DBNet, self).__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


# ===== Tiền xử lý ảnh =====
def preprocess_image(image_path, img_size=640):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"⚠️ Không thể đọc ảnh: {image_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (img_size, img_size))
    image_tensor = transforms.ToTensor()(image).unsqueeze(0)
    return image_tensor, image


# ===== Hậu xử lý: Threshold và tìm contour =====
def postprocess(prob_map, image, box_thresh=0.5, min_area=100):
    h, w = image.shape[:2]
    binary = (prob_map * 255).astype(np.uint8)
    _, thresh = cv2.threshold(binary, int(box_thresh * 255), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    return result


# ===== Hàm chính detect text =====
def detect_text(image_path):
    print(f"🔍 Đang xử lý ảnh: {image_path}")

    # Load mô hình
    model = DBNet()
    model.eval()

    # Load và xử lý ảnh
    input_tensor, raw_image = preprocess_image(image_path)

    # Dự đoán bản đồ xác suất
    with torch.no_grad():
        prob_map = model(input_tensor)[0][0].cpu().numpy()

    # Hậu xử lý và hiển thị kết quả
    result_img = postprocess(prob_map, raw_image)
    cv2.imshow("Detected Text", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ===== Chạy chương trình =====
if __name__ == "__main__":
    detect_text(r"E:\HocTap\KLTN\bien_hieu10.png")
