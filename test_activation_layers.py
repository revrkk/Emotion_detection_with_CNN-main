import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Model, model_from_json

# Assume 'model' is your pre-trained CNN model
# and 'video_stream' is your video stream capture object
# Select the layers from which you want to visualize the activations
layer_names = ['conv2d_1', 'conv2d_2', 'conv2d_3']  # Example layer names

json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Create models that output the activations of the selected layers
output_folder = "output"
activation_models = [Model(inputs=model.input, outputs=model.get_layer(name).output) for name in layer_names]
cap = cv2.VideoCapture("sample/emotion_sample.mp4")

# Get the weights of the first convolutional layer
conv1_weights = model.get_layer('conv2d_1').get_weights()[0]

# Display the weights of the first convolutional layer
plt.figure(figsize=(20, 10))
plt.suptitle('Weights of the First Convolutional Layer')
for i in range(min(32, conv1_weights.shape[3])):
    plt.subplot(4, 8, i + 1)
    plt.imshow(conv1_weights[:, :, 0, i], cmap='gray')
    plt.axis('off')
plt.savefig(os.path.join(output_folder, 'conv1_weights.png'))
plt.close()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    
    frame = cv2.resize(frame, (48, 48))
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.expand_dims(gray_frame, axis=-1)
    frame = np.expand_dims(frame, axis=0)
    
    # Get the activations for each layer for the frame
    activations = [activation_model.predict(frame) for activation_model in activation_models]
    
    # Display each layer output as an image
    for layer_name, layer_output in zip(layer_names, activations):
        num_filters = layer_output.shape[-1]
        rows = int(np.ceil(num_filters / 8))
        fig, axarr = plt.subplots(rows, 8, figsize=(20, 10))
        fig.suptitle(f"Layer: {layer_name}")
        
        for i in range(num_filters):
            ax = axarr[i // 8, i % 8] if num_filters > 8 else axarr[i]
            ax.imshow(layer_output[0, :, :, i], cmap='viridis')
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, f'{layer_name}_output.png'))
        plt.close()