import cv2
import numpy as np

# Load YOLOv4-tiny model
net = cv2.dnn.readNet("Yolo-tiny\yolov4-tiny.weights", "Yolo-tiny\yolov4-tiny.cfg")

# Load the class labels YOLOv4-tiny was trained on
with open("Yolo-tiny\coco.names", "r") as f:
    CLASSES = f.read().strip().split("\n")

# Generate random colors for each class
np.random.seed(42)
COLORS = np.random.uniform(255, 255, size=(len(CLASSES), 3))

# Initialize the video stream
cap = cv2.VideoCapture(0)  # 0 for the default camera

if not cap.isOpened():
    raise ValueError("Unable to open video source")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to a smaller size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
    # print(small_frame.shape)

    (h, w) = small_frame.shape[:2]

    # Preprocess the frame for the network
    blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # Get the output layer names
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()

    # Check if unconnected_out_layers is a list of scalars
    if np.ndim(unconnected_out_layers) == 1:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

    # Perform forward pass and obtain the detections
    detections = net.forward(output_layers)

    # Initialize lists to hold detection results
    boxes = []
    confidences = []
    class_ids = []

    # Loop over the detections
    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
            if confidence > 0.1:
                # Scale the bounding box coordinates back to the size of the image
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                # Compute the (x, y)-coordinates of the top-left corner of the bounding box
                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))

                boxes.append([startX, startY, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

    # Draw the bounding boxes and labels on the frame
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Scale the bounding box coordinates back to the size of the original frame
            (x, y) = (boxes[i][0] * 1, boxes[i][1] * 1)
            (w, h) = (boxes[i][2] * 1, boxes[i][3] * 1)

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(CLASSES[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    # frame = cv2.resize(frame, (800, 800))
    cv2.imshow('Real-Time Object', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
