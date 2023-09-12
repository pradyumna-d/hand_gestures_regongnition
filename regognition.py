import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the saved model
model = tf.keras.models.load_model('C:/Users/Admin/Desktop/ssd model/model')

# Define the hand gesture labels
gesture_labels = ['01_palm','02_L','03_fist','04_fist_moved','05_thumb','06_index','07_ok','08_palm_moved','09_C','10_down']

# Define the video capture
video_capture = cv2.VideoCapture(0)  # 0 represents the default webcam

# Define the expected input size of the model
input_width, input_height = 224, 224

while True:
    # Read the video frame
    ret, frame = video_capture.read()
    
    # Preprocess the frame
    # Resize the frame to match the input size of the model
    frame_resized = cv2.resize(frame, (input_width, input_height))
    
    # Convert the frame to RGB color
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Convert the frame to float32 and normalize
    frame_normalized = frame_rgb.astype(np.float32) / 255.0
    
    # Add batch dimension to the frame
    frame_input = np.expand_dims(frame_normalized, axis=0)
    
    # Make predictions
    predictions = model.predict(frame_input)
    
    # Get the predicted gesture label
    predicted_label = gesture_labels[np.argmax(predictions)]
    
    # Display the predicted label on the frame
    cv2.putText(frame_resized, predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Convert the frame to PIL image format
    frame_pil = Image.fromarray(np.uint8(frame_resized))
    
    # Show the frame
    frame_pil.show()
    
    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
