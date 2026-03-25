import cv2 as cv
import numpy as np
import winsound  # <--- Import winsound for beeps

# Load DNN face detector model
modelFile = r"D:\CSPROJECTS\res10_300x300_ssd_iter_140000.caffemodel"
configFile = r"D:\CSPROJECTS\deploy.prototxt"
net = cv.dnn.readNetFromCaffe(configFile, modelFile)

# Start webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera.")
    exit()

prev_center = None
multi_face_count = 0
no_face_frames = 0

# Movement threshold in pixels
movement_threshold = 50

# No face detection threshold (in frames)
no_face_threshold = 10

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    (h, w) = frame.shape[:2]
    blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0,
                                (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    # Loop over the detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        # Confidence threshold
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))
            cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

    # No face detected
    if len(faces) == 0:
        no_face_frames += 1
        if no_face_frames >= no_face_threshold:
            cv.rectangle(frame, (30, 30), (500, 100), (0, 0, 255), -1)
            cv.putText(frame, "ALERT: No face detected!", (40, 80),
                       cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
            print("WARNING: No face detected!")
            winsound.Beep(1000, 500)  # Beep for no face
    else:
        no_face_frames = 0

    # Multiple faces
    if len(faces) > 2:
        multi_face_count += 2
        cv.rectangle(frame, (30, 110), (500, 180), (0, 0, 255), -1)
        cv.putText(frame, "ALERT: Multiple faces detected!", (40, 160),
                   cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
        print(f"WARNING: Multiple faces detected! ({multi_face_count} times)")
        winsound.Beep(1200, 500)  # Beep for multiple faces

        if multi_face_count > 2:
            print("Proctoring stopped due to repeated multiple faces.")
            break

    # Face movement
    if len(faces) == 1:
        (x, y, w_box, h_box) = faces[0]
        current_center = (x + w_box // 2, y + h_box // 2)

        if prev_center is not None:
            dx = abs(current_center[0] - prev_center[0])  # Calculate the change in x
            dy = abs(current_center[1] - prev_center[1])  # Calculate the change in y
            if dx > movement_threshold or dy > movement_threshold:
                cv.rectangle(frame, (30, 190), (600, 260), (0, 0, 255), -1)
                cv.putText(frame, "ALERT: Face movement detected!", (40, 240),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
                print("WARNING: Face moved too much!")
                winsound.Beep(1500, 500)  # Beep for movement

        prev_center = current_center

    cv.imshow("Proctoring System (DNN)", frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        print("Proctoring stopped by user.")
        break

cap.release()
cv.destroyAllWindows()




