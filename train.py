from ultralytics import YOLO

model = YOLO("yolov8m.pt")   # medium — better than small for small objects

model.train(
    data="dataset/data.yaml",
    epochs=50,               # minimum for good generalization
    imgsz=1024,              # keep — helps far objects
    batch=2,                 # keep if low VRAM
    name="chicken_model",
    patience=15,             # stops early if no improvement
    augment=True,            # random flips, zoom, brightness
    hsv_h=0.015,             # color variation — handles lighting changes
    hsv_v=0.4,               # brightness variation — dawn/dusk
    scale=0.5,               # random zoom out — simulates far chickens
    fliplr=0.5,              # random horizontal flip
)