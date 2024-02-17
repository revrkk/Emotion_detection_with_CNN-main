import cv2
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import matplotlib.pyplot as plt

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}


def display_layer_activations(model, input_image, layerName):
    # Assume 'model' is your pre-trained CNN model
    # and 'input_image' is the input image you want to use
    # Select the layer from which you want to visualize the activations
    layer_name = layerName  # Example layer name

    # Create a model that outputs the activations of the selected layer
    activation_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer(layer_name).output)

    # Preprocess the input image (assuming it's in BGR format)
    input_image = cv2.resize(input_image, (model.input_shape[1], model.input_shape[2]))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    # Get the activations for the input image
    activations = activation_model.predict(input_image)

    # Display each filter output as an image
    num_filters = activations.shape[-1]
    rows = int(np.ceil(num_filters / 8))  # Assuming 8 columns for display
    fig, axarr = plt.subplots(rows, 8, figsize=(20, 10))

    for i in range(num_filters):
        ax = axarr[i // 8, i % 8] if num_filters > 8 else axarr[i]
        ax.imshow(activations[0, :, :, i], cmap='viridis')  # Display the activation
        ax.axis('off')

    plt.tight_layout()
    plt.show()


# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
#cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
cap = cv2.VideoCapture("sample/emotion_sample.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)

        for layer in emotion_model.layers:
            # Create a new model that outputs the layer's output
            intermediate_layer_model = tf.keras.Model(inputs=emotion_model.input, outputs=layer.output)
            # Get the output of the current layer for the input image
            #intermediate_output = intermediate_layer_model.predict(cropped_img)
            display_layer_activations(intermediate_layer_model, cropped_img, layer.name)
            # Print the output shape
            #print(f"Output shape of {layer.name}: {intermediate_output.shape}")
        
        maxindex = int(np.argmax(emotion_prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
