from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train/weights/best.pt")
# Test image
results = model("image1.jpg")

# Show result
results[0].show()

# Save output
results[0].save("output.jpg")

print("Detection completed")