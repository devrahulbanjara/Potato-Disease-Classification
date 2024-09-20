import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("/home/rahul/Documents/Potato-Disease-Classification/models/classifier_model.keras")

# Class names for potato diseases
class_names = ['Early_blight', 'Healthy', 'Late_blight']

# Define a function to preprocess the image
def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((256, 256))  # Assuming your model expects 256x256 images
    img = img.convert("RGB")
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Create the Streamlit app
def main():
    st.title("Potato Disease Classifier")

    uploaded_file = st.file_uploader("Upload an image of a potato plant")

    if uploaded_file is not None:
        image = preprocess_image(uploaded_file)
        prediction = model.predict(image)
        predicted_class = np.argmax(prediction)

        st.write("Predicted class:", class_names[predicted_class])

if __name__ == "__main__":
    main()