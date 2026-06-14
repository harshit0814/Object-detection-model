from ultralytics import YOLO
import cv2
# Load pre-trained YOLOv8 model
model = YOLO("yolov8n.pt")
# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot access webcam")
    exit()
while True:
    # Read frame
    ret, frame = cap.read()
    if not ret:
        break
    # Object Detection
    results = model(frame)
    # Draw bounding boxes
    annotated_frame = results[0].plot()
    # Display output
    cv2.imshow("Real-Time Object Detection", annotated_frame)
    # Press q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()