import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("Yolo-tiny/yolov4-tiny.weights", "Yolo-tiny/yolov4-tiny.cfg")
classes = []
with open("Yolo-tiny/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
try:
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
except Exception:
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# Load an image
image_path = "ryoji-iwata-dlBXwGlzfcs-unsplash.jpg"  # Replace with your image path
frame = cv2.imread(image_path)

if frame is None:
    raise ValueError("Unable to open image")

# Resize the image
small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
(h, w) = small_frame.shape[:2]

# Detecting objects
blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
net.setInput(blob)
detections = net.forward(output_layers)

# Information to display
class_ids = []
confidences = []
boxes = []

for out in detections:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            # Object detected
            center_x = int(detection[0] * w)
            center_y = int(detection[1] * h)
            dw = int(detection[2] * w)
            dh = int(detection[3] * h)

            # Rectangle coordinates
            x = int(center_x - dw / 2)
            y = int(center_y - dh / 2)

            boxes.append([x, y, dw, dh])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# Non-max suppression
indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

# Draw bounding box with label and confidence
for i in indices.flatten():
    box = boxes[i]
    x, y, w, h = box[0], box[1], box[2], box[3]

    label = str(classes[class_ids[i]])
    confidence = confidences[i]
    color = [int(c) for c in np.random.uniform(0, 255, 3)]
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{label}: {confidence:.2f}", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Display the image
cv2.imshow('Image', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()