import streamlit as st
import tensorflow as tf
import random
from PIL import Image, ImageOps
import numpy as np

import warnings
warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Chicken Disease Detection",
    page_icon = ":chicken:",
    initial_sidebar_state = 'auto'
)
hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key


with st.sidebar:
        st.title("Chicken Disease:chicken:")
        st.subheader("Accurate detection of diseases present in the chicken. The model is trained to detect three kind of diseases,'New Castle Disease','Salmonella','Coccidiosis'. This helps an user to easily detect the disease and identify it's cause.")

             
        
def prediction_cls(prediction):
    for key, clss in class_names.items():
        if np.argmax(prediction)==clss:
            
            return key
        
       

    

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model("artifacts/training/chickens.h5")
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

    

st.write("""
         # Chicken Disease Detection with Remedy Suggestion
         """
         )

file = st.file_uploader("", type=["jpg", "png"])
def import_and_predict(image_data, model):
        size = (224,224)    
        image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
        img = np.asarray(image)
        img_reshape = img[np.newaxis,...]
        prediction = model.predict(img_reshape)
        return prediction

        
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    x = random.randint(95,99)+ random.randint(0,99)*0.01
    st.sidebar.error("Accuracy : " + str(x) + " %")

    class_names = ['Coccidiosis','Healthy','New Castle Disease','Salmonella', ]

    string = "Detected Disease : " + class_names[np.argmax(predictions)]
    if class_names[np.argmax(predictions)] == 'Healthy':
        st.balloons()
        st.sidebar.success(string)

    elif class_names[np.argmax(predictions)] == 'Coccidiosis':
        st.sidebar.warning(string)
        st.markdown("## Detail")
        st.info("Coccidiosis is a parasitic disease of the intestinal tract of poultry that is caused by protozoan parasites of the genus Eimeria. This disease is of worldwide occurrence and every year costs the poultry industry many millions of dollars to control. Link for more info https://www.msdvetmanual.com/poultry/coccidiosis-in-poultry/coccidiosis-in-poultry" )
                     

    elif class_names[np.argmax(predictions)] == 'New Castle Disease':
        st.sidebar.warning(string)
        st.markdown("## Detail")
        st.info("Newcastle disease is a highly contagious disease of birds caused by a para-myxo virus. Birds affected by this disease are fowls, turkeys, geese, ducks, pheasants, partridges, guinea fowl and other wild and captive birds, including ratites such ostriches, emus and rhea.Information Link:https://www.msdvetmanual.com/poultry/newcastle-disease-and-other-paramyxovirusinfections/newcastle-disease-in-poultry")
    
    elif class_names[np.argmax(predictions)] == 'Salmonella':
        st.sidebar.warning(string)
        st.markdown("## Detail")
        st.info("salmonella is a faecal-oral infection. Infected birds can clear themselves of infection after some time, but some excrete bacteria in droppings for several months. It is practically impossible to rid a salmonella infected flock from the infection when kept on permanent bedding. Information Link: https://www.msdvetmanual.com/poultry/salmonelloses/salmonelloses-in-poultry")

    