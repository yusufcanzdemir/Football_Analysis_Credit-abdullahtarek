from ultralytics import YOLO

model = YOLO('models/best.pt')

results = model.predict('video.mp4',save = True)

print(results[0])
print("_____________________")

for box in results[0].boxes:
    print(box)