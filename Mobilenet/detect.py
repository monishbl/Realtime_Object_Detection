import cv2
import numpy as np

# Load the MobileNet SSD model
net = cv2.dnn.readNetFromCaffe('Mobilenet\deploy.prototxt', 'Mobilenet\mobilenet_iter_73000.caffemodel')

# Load the class labels MobileNet SSD was trained on
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "table", "cow",
           "diningtable", "dog", "horse", "man", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Generate random colors for each class
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Initialize the video stream
cap = cv2.VideoCapture(0)  # 0 for the default camera

if not cap.isOpened():
    raise ValueError("Unable to open video source")

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    # Preprocess the frame for the network
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)

    # Perform forward pass and obtain the detections
    detections = net.forward()

    # Loop over the detections
    for i in range(detections.shape[2]):
        # Extract the confidence (probability) associated with the prediction
        confidence = detections[0, 0, i, 2]

        # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
        if confidence > 0.2:
            # Extract the index of the class label from the detection
            idx = int(detections[0, 0, i, 1])

            # Compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # Draw the bounding box around the detected object
            label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    # Display the resulting frame
    # frame = cv2.resize(frame, (800, 800))
    cv2.imshow('Real-Time Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
