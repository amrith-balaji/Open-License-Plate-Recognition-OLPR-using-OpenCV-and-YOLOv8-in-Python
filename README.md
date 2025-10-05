# ANPR-YOLO-EasyOCR

Automatic Number Plate Recognition (ANPR) for Indian vehicles using YOLOv8 and EasyOCR.

This project detects license plates in images and reads their text. It uses:
- **YOLOv8** for detecting plates
- **EasyOCR** for reading the text
- Image preprocessing and post-processing to improve OCR results

---

## ⚙️ Requirements

Install the required Python packages:

```bash
pip install -r requirements.txt
```
Note: If you have a CUDA GPU, set gpu=True in detect_and_read.py to accelerate OCR.

---

## 🏗 Dataset

We use the Car License Plate dataset by Matheus Santos Almeida on Roboflow.

Steps to Download:

1. Go to the dataset link: https://universe.roboflow.com/matheus-santos-almeida/car_license_plate

2. Click “Download Dataset”

3. Choose format: YOLOv8 PyTorch

4. Extract the downloaded dataset into the dataset/ folder

---

## 🏋️ Training YOLOv8

Train your YOLOv8 model on the dataset:
``` bash
yolo detect train data=dataset/data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

* Output weights will be saved in runs/detect/train/weights/best.pt.

* Copy best.pt into models/ folder.

---

## 🖼 Testing / Detection

Place any test images in input_images/ and run:

python src/detect_and_read.py --image input_images/car1.jpg


* Annotated images will be saved in output_images/

* Console prints detected plates and bounding boxes

---

## 🧩 Notes

* The current OCR uses EasyOCR, which is generic. Accuracy may vary depending on plate font, color, or lighting.

You can improve recognition by:

* Adding more training data

* Fine-tuning OCR models on Indian license plates

* Implementing sequence recognition models

---

## 💻 Requirements Recap

* Python 3.10+ (Python 3.13 works, but GUI in OpenCV may fail — we use matplotlib for display)

* GPU is optional, but speeds up OCR significantly

* Packages installed via requirements.txt

--- 

## 🔧 How to Add Your Own Images

1. Place images in input_images/

2. Run detection script:
  ```bash
    python src/detect_and_read.py --image input_images/your_image.jpg
  ```

3. Check output_images/ for annotated results
