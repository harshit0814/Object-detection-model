from ultralytics import YOLO
import cv2
print("Loading model...")
# Use default YOLO model
model = YOLO("yolov8n.pt")
print("Model loaded")
# Detect image
results = model("image1.jpg")
# Draw boxes
annotated_img = results[0].plot()
# Save output
cv2.imwrite("output.jpg", annotated_img)
# Show output
cv2.imshow("Detection", annotated_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Detection Completed")