import numpy as np


def get_model(name, pretrained=True, trainable=True):
    if name == "xception": 
        if pretrained: 
            from models.xception import modify_Xception
            model = modify_Xception(trainable=trainable)
        else: 
            from keras.applications import Xception
            model = Xception(include_top=False, weights="None", input_shape=(512, 512, 1))

    elif name == "mobilenet": 
        if pretrained: 
            from models.mobilenet import modify_MobileNet
            model = modify_MobileNet(trainable=trainable)
        else: 
            from keras.applications import MobileNetV3Small
            model = MobileNetV3Small(include_top=False, weights="None", input_shape=(512, 512, 1), include_preprocessing=False)
        
    else:
        raise NotImplementedError(f"'{name}' model name not implemented!")
    
    return model

def get_preprocessing_func(name):
    if name == "xception":
        from keras.applications.xception import preprocess_input
        
    elif name == "mobilenet":
        from keras.applications.mobilenet_v3 import preprocess_input
        
    else:
        raise NotImplementedError(f"'{name}' model name not implemented!")
        
    return preprocess_input

def avg_weights(weights):
    avg = np.mean(weights, axis=2).reshape(weights[:, :, -1:, :].shape)
    return avg
