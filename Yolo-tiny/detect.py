import cv2
import numpy as np

net = cv2.dnn.readNet("Yolo-tiny\yolov4-tiny.weights", "Yolo-tiny\yolov4-tiny.cfg")

with open("Yolo-tiny\coco.names", "r") as f:
    CLASSES = f.read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.uniform(255, 255, size=(len(CLASSES), 3))

cap = cv2.VideoCapture(0)  
if not cap.isOpened():
    raise ValueError("Unable to open video source")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)

    (h, w) = small_frame.shape[:2]

    blob = cv2.dnn.blobFromImage(small_frame, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()

    if np.ndim(unconnected_out_layers) == 1:
        output_layers = [layer_names[i - 1] for i in unconnected_out_layers]
    else:
        output_layers = [layer_names[i[0] - 1] for i in unconnected_out_layers]

    detections = net.forward(output_layers)

    boxes = []
    confidences = []
    class_ids = []

    for output in detections:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.1:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))

                boxes.append([startX, startY, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.2)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0] * 1, boxes[i][1] * 1)
            (w, h) = (boxes[i][2] * 1, boxes[i][3] * 1)

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(CLASSES[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Real-Time Object', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
