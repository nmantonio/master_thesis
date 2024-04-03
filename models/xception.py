from keras.models import Model
from keras.applications import Xception

from models.utils import avg_weights


def modify_Xception(trainable=True):
    original_model = Xception(include_top=False, weights="imagenet", input_shape=(512, 512, 3))

    # Get model config
    model_config = original_model.get_config()

    # Modify input layer to require only one-channel images
    model_config["layers"][0]["config"]["batch_input_shape"] = (None, 512, 512, 1)

    # Create modified model
    model = Model.from_config(model_config)

    layer_names = [model_config["layers"][x]["name"] for x in range(len(model_config["layers"]))]
    first_conv_name = model_config["layers"][1]["name"]

    # Change weights to the pretrained ones (avg and imagenet)
    for layer in original_model.layers:
        if layer.name in layer_names: 
            
            # If the layer have weights
            if layer.get_weights() != []:
                target_layer = model.get_layer(layer.name)

                # Modify first conv layer to deal with modified input shape
                if layer.name == first_conv_name: 
                    ## Get original weights
                    weights = layer.get_weights()[0]
                    
                    ## Get new weights by averaging the three in_channels (other options possible)
                    new_weights = avg_weights(weights)
                    
                    ## Set new weights
                    if not model_config["layers"][1]["config"]["use_bias"]:
                        target_layer.set_weights([new_weights])
                    else:
                        biases = layer.get_weights()[1]
                        target_layer.set_weights([new_weights, biases])
                        
                    target_layer.trainable = trainable
                    
                else: 
                    target_layer.set_weights(layer.get_weights())
                    target_layer.trainable = trainable

                # target_layer.trainable = False
                
    return model

# model = modify_Xception()
# model.summary()