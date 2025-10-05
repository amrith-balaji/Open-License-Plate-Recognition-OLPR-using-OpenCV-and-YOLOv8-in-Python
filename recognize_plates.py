import cv2
import matplotlib.pyplot as plt
import easyocr
from ultralytics import YOLO
import re
import numpy as np

# Load models
detector = YOLO("runs/detect/train/weights/best.pt")  # your trained YOLO model
reader = easyocr.Reader(['en'], gpu=False)  # set gpu=True if you have CUDA

# --- Helper functions ---

def preprocess_plate(crop):
    """Enhance cropped license plate for OCR."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 11, 17, 17)
    _, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Resize for better OCR accuracy
    h, w = thresh.shape
    new_w = 400
    new_h = int(h * (new_w / w))
    resized = cv2.resize(thresh, (new_w, new_h))
    return resized

def clean_plate_text(text):
    """Post-process OCR text to match Indian plate patterns."""
    text = text.replace(" ", "").upper()
    # Common OCR fixes
    text = text.replace("O", "0").replace("I", "1").replace("S", "5").replace("Z", "2")
    # Match Indian plate pattern
    pattern = r'[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}'
    match = re.search(pattern, text)
    return match.group(0) if match else text

# --- Main detection function ---

def detect_and_read(image_path, conf_thresh=0.3):
    img = cv2.imread(image_path)
    results = detector(img, conf=conf_thresh)

    outputs = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_processed = preprocess_plate(crop)
            ocr_res = reader.readtext(crop_processed, detail=0)
            text_raw = " ".join(ocr_res).strip()
            text = clean_plate_text(text_raw)

            outputs.append({
                'bbox': (x1, y1, x2, y2),
                'text': text
            })

            # Annotate original image
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, text, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    # Display annotated image using matplotlib
    plt.figure(figsize=(12,8))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return outputs

# --- Run example ---

if __name__ == "__main__":
    image_path = "test_image.jpg"  # replace with your image path
    results = detect_and_read(image_path)

    print("✅ Detected license plates:")
    for plate in results:
        print(f"{plate['text']} at {plate['bbox']}")

