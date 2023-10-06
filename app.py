import os
import streamlit as st
from PIL import Image
from src.ChickenDisease.utils.common import decodeImage
from src.ChickenDisease.pipeline.predict import PredictionPipeline

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

class ClientApp:
    def __init__(self):
        self.filename = "inputImage.jpg"
        self.classifier = PredictionPipeline(self.filename)

clApp = ClientApp()

st.title("Chicken Disease Classification")

# Create a sidebar for training
if st.sidebar.button("Train Model"):
    os.system("python main.py")
    st.sidebar.success("Training done successfully!")

# Create the main content
st.sidebar.header("Upload Image")
image = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg","png"])

if image:
    try:
        # Open the uploaded image with PIL to check if it's a valid image
        with Image.open(image) as img:
            st.image(img, caption="Uploaded Image", use_column_width=True)
            st.sidebar.success("Image uploaded successfully!")

            # Make a prediction if an image is uploaded
            if st.sidebar.button("Predict"):
                with st.spinner("Predicting..."):
                        try:
                            image_bytes= img.tobytes()
                            
                                # Decode and process the image
                            decodeImage(image_bytes, clApp.filename)
                                # Make a prediction
                            result = clApp.classifier.predict()
                                
                                # Display the prediction
                            st.success("Prediction:")
                            st.json(result)
                        
                        except Exception as e:
                            st.error(f"Error: {e}")

                st.success("Prediction:")
                st.json(result)
    except Exception as e:
        st.error(f"Error: {e}")
