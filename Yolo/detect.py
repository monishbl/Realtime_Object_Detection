import cv2
import numpy as np
import os
import requests

# Define the path to the weights file
weights_path = "Yolo/yolov4.weights"

# Check if the weights file exists
if not os.path.exists(weights_path):
    # Define the URL of the weights file in your Hugging Face repository
    url = "https://cdn-lfs-us-1.huggingface.co/repos/9f/fe/9ffe91d46cfcedc00334199261c9254a8f99615d8fca52412bfd18ff67b7f632/e8a4f6c62188738d86dc6898d82724ec0964d0eb9d2ae0f0a9d53d65d108d562?response-content-disposition=attachment%3B+filename*%3DUTF-8%27%27yolov4.weights%3B+filename%3D%22yolov4.weights%22%3B&Expires=1718554556&Policy=eyJTdGF0ZW1lbnQiOlt7IkNvbmRpdGlvbiI6eyJEYXRlTGVzc1RoYW4iOnsiQVdTOkVwb2NoVGltZSI6MTcxODU1NDU1Nn19LCJSZXNvdXJjZSI6Imh0dHBzOi8vY2RuLWxmcy11cy0xLmh1Z2dpbmdmYWNlLmNvL3JlcG9zLzlmL2ZlLzlmZmU5MWQ0NmNmY2VkYzAwMzM0MTk5MjYxYzkyNTRhOGY5OTYxNWQ4ZmNhNTI0MTJiZmQxOGZmNjdiN2Y2MzIvZThhNGY2YzYyMTg4NzM4ZDg2ZGM2ODk4ZDgyNzI0ZWMwOTY0ZDBlYjlkMmFlMGYwYTlkNTNkNjVkMTA4ZDU2Mj9yZXNwb25zZS1jb250ZW50LWRpc3Bvc2l0aW9uPSoifV19&Signature=EY3FliLIe47088a9m1wlxC3Ln80G5ZkIMlhoZaa3Ei87AvVGoqApyBoLRz9UCfVUHXiXDTTAdy3KNYZLbqwcaVgc0ifiR5R7vcLb8CXYSvt9CEmOlqY5OTFHOakxVnCiU%7EWAQ2hGUZgBZtI17QdAZ29LXA368sRMQGyzSLVsJ8LSpHlpDImTkAxCO-7sjlzIbammJnWRW6dUV7mb6ByA7sdrWtZayENpKYkqORRiZCErPgoeaNWSi0jd%7E7XD7pDrQ4t1PrbPt1SVMyVyL624-ERBloQ%7E4wgsUL6set8K4bJJsojU8xuR0q0Jl1DIjgY-liIdGPeMAkmMpXAacQ4zhQ__&Key-Pair-Id=K2FPYV99P2N66Q"

    # Send a GET request to the URL
    response = requests.get(url)

    # Ensure the request was successful
    response.raise_for_status()

    # Write the content of the response to a file in the parent directory
    with open(weights_path, "wb") as f:
        f.write(response.content)
# Load YOLOv4 model
net = cv2.dnn.readNet(weights_path, "Yolo\yolov4.cfg")

# Load the class labels YOLOv4 was trained on
with open("Yolo\coco.names", "r") as f:
    CLASSES = f.read().strip().split("\n")

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

    # Resize the frame to a smaller size for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

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
            if confidence > 0.5:
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
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)

    # Draw the bounding boxes and labels on the frame
    if len(idxs) > 0:
        for i in idxs.flatten():
            # Scale the bounding box coordinates back to the size of the original frame
            (x, y) = (boxes[i][0] * 2, boxes[i][1] * 2)
            (w, h) = (boxes[i][2] * 2, boxes[i][3] * 2)

            color = [int(c) for c in COLORS[class_ids[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(CLASSES[class_ids[i]], confidences[i])
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the resulting frame
    # frame = cv2.resize(frame, (800, 800))
    cv2.imshow('Real-Time Object Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()