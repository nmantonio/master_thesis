import numpy as np
import os
from paths import *
from keras.src import backend


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
            
    elif name == "densenet":
        if pretrained: 
            from models.densenet import modify_densenet
            model = modify_densenet(trainable=trainable)
        else: 
            from keras.applications import DenseNet201
            model = DenseNet201(include_top=False, weights="None", input_shape=(512, 512, 1))
        
    else:
        raise NotImplementedError(f"'{name}' model name not implemented!")
    
    return model

def get_preprocessing_func(name, pretrained=None, database=None, task=None):
    if name == "xception":
        if database == DATABASE_PATH:
            from keras.applications.xception import preprocess_input
        elif database == CROPPED_DATABASE_PATH:
            def preprocess_input(x, mask=None):
                x = x.astype(backend.floatx(), copy=False)
                x /= 127.5
                if mask is not None:
                    x[mask > 0] = x[mask > 0] - 1
                else:
                    x = x - 1
                return x
        else: 
            raise ValueError("Wrong database path (preprocessing func)")
        
    elif name == "mobilenet":
        if database == DATABASE_PATH or database == CROPPED_DATABASE_PATH:
            def preprocess_input(x, mask=None):
                x = x.astype(backend.floatx(), copy=False)
                x /= 127.5
                if mask is not None:
                    x[mask > 0] = x[mask > 0] - 1
                else:
                    x = x - 1
                return x
        else: 
            raise ValueError("Wrong database path (preprocessing func)")
        
    elif name == "densenet":

        if pretrained: 
            mean = np.mean([0.485, 0.456, 0.406])
            std = np.mean([0.229, 0.224, 0.225])
            
        else: 
            if task == "detection":
                if database==DATABASE_PATH:
                    mean = np.array([0.496])
                    std = np.array([0.257])
                elif database==CROPPED_DATABASE_PATH:
                    mean = np.array([0.438])
                    std = np.array([0.160]) 
            elif task == "classification":
                if database==DATABASE_PATH:
                    mean = np.array([0.501])
                    std = np.array([0.257])
                elif database==CROPPED_DATABASE_PATH:
                    mean = np.array([0.456])
                    std = np.array([0.162]) 
            else: 
                raise ValueError(f"{task} not available! (PREPROCESSING FUNC SELECTION)") 
            
        def preprocess_input(x, mask=None):
            x = x.astype(backend.floatx(), copy=False)
            x /= 255.0
            if mask is not None:
                x[mask > 0] = (x[mask > 0] - mean)/std
            else:
                x = (x - mean)/std
            return x

    else:
        raise NotImplementedError(f"'{name}' model name not implemented!")
        
    return preprocess_input

def avg_weights(weights):
    avg = np.mean(weights, axis=2).reshape(weights[:, :, -1:, :].shape)
    return avg

def plot_model(model_name):
    model = get_model(model_name)
    from keras.utils import plot_model
    img_name = model_name + ".png"
    plot_model(model, to_file=img_name)
