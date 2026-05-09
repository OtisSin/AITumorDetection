import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import numpy as np
from pathlib import Path


class TumorDetector:
    def __init__(self, model_path='best_faster_rcnn.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Инициализация детектора на: {self.device}")

        # Создаём архитектуру
        model = fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)

        # Загружаем веса
        if not Path(model_path).exists():
            raise FileNotFoundError(f"❌ Веса модели не найдены: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        self.model = model
        print("✅ Модель успешно загружена в память!")

    def detect(self, image, conf_threshold=0.5):
        """
        Принимает numpy-массив (RGB от Gradio или BGR от OpenCV)
        Возвращает: (result_image_rgb, report_text)
        """
        # Gradio передаёт RGB, OpenCV работает с BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image

        h_orig, w_orig, _ = img_bgr.shape
        img_resized = cv2.resize(img_bgr, (640, 640))
        img_tensor = torch.from_numpy(img_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        img_tensor = img_tensor.to(self.device)

        # Инференс
        with torch.no_grad():
            predictions = self.model(img_tensor)

        boxes = predictions[0]['boxes'].cpu().numpy()
        scores = predictions[0]['scores'].cpu().numpy()
        labels = predictions[0]['labels'].cpu().numpy()

        scale_x = w_orig / 640.0
        scale_y = h_orig / 640.0
        detected = False
        report_lines = []
        result_image = img_bgr.copy()

        for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
            if score < conf_threshold:
                continue

            x1, y1, x2, y2 = box
            x1_o, y1_o = int(x1 * scale_x), int(y1 * scale_y)
            x2_o, y2_o = int(x2 * scale_x), int(y2 * scale_y)

            cx, cy = (x1_o + x2_o) // 2, (y1_o + y2_o) // 2
            w, h = x2_o - x1_o, y2_o - y1_o

            report_lines.append(f"🔹 Опухоль #{i + 1} | Уверенность: {score * 100:.1f}%")
            report_lines.append(f"   Центр: ({cx}, {cy}) | Размер: {w}×{h} px\n")

            color = (0, 255, 0) if score > 0.8 else (0, 255, 255)
            cv2.rectangle(result_image, (x1_o, y1_o), (x2_o, y2_o), color, 3)
            cv2.putText(result_image, f"Tumor: {score * 100:.1f}%", (x1_o, y1_o - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            detected = True

        if not detected:
            report = "⚠️ Опухоль не обнаружена.\nУверенность ниже порога."
            cv2.putText(result_image, "No Tumor", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        else:
            report = "\n".join(report_lines).strip()

        # Возвращаем в RGB для совместимости с Gradio/веб-браузерами
        result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        return result_rgb, report


# Глобальный экземпляр (загружается один раз при импорте)
detector = TumorDetector()


def process_image(image, conf_threshold=0.5):
    """Обёртка для удобного импорта в AI_App.py"""
    if image is None:
        return None, "❌ Загрузите изображение МРТ."
    return detector.detect(image, conf_threshold)


# Тест при прямом запуске
if __name__ == '__main__':
    test_img = cv2.imread('mri_scan.jpg')
    if test_img is not None:
        res, rep = detector.detect(test_img, 0.5)
        cv2.imwrite('result_test.jpg', res)
        print(rep)