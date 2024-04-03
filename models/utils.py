import numpy as np


def modify_model(name, trainable=True):
    if name == "xception": 
        from models.xception import modify_Xception as model

    elif name == "mobilenet": 
        from models.mobilenet import modify_MobileNet as model
        
    else:
        raise NotImplementedError(f"'{name}' model name not implemented!")
    
    return model(trainable=trainable)

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
