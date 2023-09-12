import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained SSD model
model = tf.keras.models.load_model('C:/Users/Admin/Desktop/ssd_model')

class_labels = ['01_palm','02_L','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_C','10_down']

# Set the input width and height
input_width, input_height = 244, 244

# Load the video feed
video = cv2.VideoCapture(0)  # Use 0 for the default camera, or specify the video file path

while True:
    # Read a frame from the video feed
    ret, frame = video.read()
    if not ret:
        break

    # Preprocess the frame for the SSD model
    resized_frame = cv2.resize(frame, (input_width, input_height))
    preprocessed_frame = np.expand_dims(resized_frame, axis=0)

    # Perform object detection
    predictions = model.predict(preprocessed_frame)

    # Loop over the predictions and draw bounding boxes
    for prediction in predictions:
        class_id = np.argmax(prediction)
        confidence = prediction[class_id]

        if confidence > 0.5:  # Adjust the confidence threshold as needed
            class_label = class_labels[class_id]

            # Get the coordinates of the predicted bounding box
            x1 = int(prediction[1] * frame.shape[1])
            y1 = int(prediction[2] * frame.shape[0])
            x2 = int(prediction[3] * frame.shape[1])
            y2 = int(prediction[4] * frame.shape[0])

            # Draw the bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all windows
video.release()
cv2.destroyAllWindows()
