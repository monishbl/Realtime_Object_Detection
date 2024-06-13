import cv2
import numpy as np
import os
import requests

weights_path = "Yolo/yolov4.weights"

if not os.path.exists(weights_path):
    url = "https://cdn-lfs-us-1.huggingface.co/repos/9f/fe/9ffe91d46cfcedc00334199261c9254a8f99615d8fca52412bfd18ff67b7f632/e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27yolov4.weights%3B+filename%3D%22yolov4.weights%22%3B&Expires=1718554556&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODU1NDU1Nn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzlmL2ZlLzlmZmU5MWQ0NmNmY2VkYzAwMzM0MTk5MjYxYzkyNTRhOGY5OTYxNWQ4ZmNhNTI0MTJiZmQxOGZmNjdiN2Y2MzIvZThhNGY2YzYyMTg4NzM4ZDg2ZGM2ODk4ZDgyNzI0ZWMwOTY0ZDBlYjlkMmFlMGYwYTlkNTNkNjVkMTA4ZDU2Mj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=EY3FliLIe47088a9m1wlxC3Ln80G5ZkIMlhoZaa3Ei87AvVGoqApyBoLRz9UCfVUHXiXDTTAdy3KNYZLbqwcaVgc0ifiR5R7vcLb8CXYSvt9CEmOlqY5OTFHOakxVnCiU%7EWAQ2hGUZgBZtI17QdAZ29LXA368sRMQGyzSLVsJ8LSpHlpDImTkAxCO-7sjlzIbammJnWRW6dUV7mb6ByA7sdrWtZayENpKYkqORRiZCErPgoeaNWSi0jd%7E7XD7pDrQ4t1PrbPt1SVMyVyL624-ERBloQ%7E4wgsUL6set8K4bJJsojU8xuR0q0Jl1DIjgY-liIdGPeMAkmMpXAacQ4zhQ__&Key-Pair-Id=K2FPYV99P2N66Q"

    response = requests.get(url)

    response.raise_for_status()

    with open(weights_path, "wb") as f:
        f.write(response.content)
net = cv2.dnn.readNet(weights_path, "Yolo\yolov4.cfg")

with open("Yolo\coco.names", "r") as f:
    CLASSES = f.read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise ValueError("Unable to open video source")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

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

            if confidence > 0.5:
                box = detection[0:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")

                startX = int(centerX - (width / 2))
                startY = int(centerY - (height / 2))

                boxes.append([startX, startY, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0] * 2, boxes[i][1] * 2)
            (w, h) = (boxes[i][2] * 2, boxes[i][3] * 2)

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(CLASSES[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imshow('Real-Time Object Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()