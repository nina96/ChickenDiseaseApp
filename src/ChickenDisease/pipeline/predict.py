import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        model = load_model(os.path.join("artifacts","training", "chickens.h5"))

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 0:
            prediction = 'Coccidiosis'
            return [{ "Disease":prediction,
                      "Detail":"Coccidiosis is a parasitic disease of the intestinal tract of poultry that is caused by protozoan parasites of the genus Eimeria. This disease is of worldwide occurrence and every year costs the poultry industry many millions of dollars to control.",
                      "Information Link": "https://www.msdvetmanual.com/poultry/coccidiosis-in-poultry/coccidiosis-in-poultry" 
                     
                     }]
        elif result[0] == 1:
            prediction = 'Healthy'
            return [{ "Disease":prediction,
                     "Information":"Your Chicken is healthy"
                     }]
        elif result[0] == 2:
            prediction = 'New Castle Disease'
            return [{ "Disease":prediction,
                     "Detail":"Newcastle disease is a highly contagious disease of birds caused by a para-myxo virus. Birds affected by this disease are fowls, turkeys, geese, ducks, pheasants, partridges, guinea fowl and other wild and captive birds, including ratites such ostriches, emus and rhea.",
                     "Information Link":"https://www.msdvetmanual.com/poultry/newcastle-disease-and-other-paramyxovirusinfections/newcastle-disease-in-poultry"}]
        
        else:
            prediction = 'Salmonella'
            return [{"Disease": prediction,
                     "Detail":"salmonella is a faecal-oral infection. Infected birds can clear themselves of infection after some time, but some excrete bacteria in droppings for several months. It is practically impossible to rid a salmonella infected flock from the infection when kept on permanent bedding",
                     "Information Link": "https://www.msdvetmanual.com/poultry/salmonelloses/salmonelloses-in-poultry"}]