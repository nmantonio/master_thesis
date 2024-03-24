import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
from data_loader import preprocess
import os
import cv2

DATABASE_PATH = r"C:\Users\tonin\Desktop\Master\TFM\PROCESSED_DATABASE"

model = load_model(r"C:\Users\tonin\Desktop\Master\TFM\xception_sgd_lowlr.h5")

# preds = model.predict(x)
# print('Predicted:', preds[0])

# pred_class = np.round(preds[0]).astype(np.uint8)[0]
# print(pred_class)


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras import Model
from PIL import Image


def show_imgwithheat(img_path, heatmap, alpha=0.4, return_array=False):
    """Show the image with heatmap.

    Args:
        img_path: string.
        heatmap: image array, get it by calling grad_cam().
        alpha: float, transparency of heatmap.
        return_array: bool, return a superimposed image array or not.
    Return:
        None or image array.
    """
    img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

    # imgwithheat = Image.fromarray(superimposed_img)
    # try:
    #     display(imgwithheat)
    # except NameError:
    #     imgwithheat.show()

    # if return_array:
    return superimposed_img

def grad_cam(model, img,
             layer_name="block14_sepconv2_act",
             category_id=None):
    """Get a heatmap by Grad-CAM.

    Args:
        model: A model object, build from tf.keras 2.X.
        img: An image ndarray.
        layer_name: A string, layer name in model.
        category_id: An integer, index of the class.
            Default is the category with the highest score in the prediction.

    Return:
        A heatmap ndarray(without color).
    """
    img_tensor = np.expand_dims(img, axis=0)

    conv_layer = model.get_layer(layer_name)
    heatmap_model = Model([model.inputs], [conv_layer.output, model.output])

    with tf.GradientTape() as gtape:
        conv_output, predictions = heatmap_model(img_tensor)
        if category_id is None:
            category_id = np.argmax(predictions[0])
        output = predictions[:, category_id]
        # print("OUTPUT: ", output)
        if output[0] < 0.5:
            print ("Predicted: abnormal")
        else: 
            print("Predicted: normal")
        grads = gtape.gradient(output, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat

    return np.squeeze(heatmap)

print("------------------------------------------------")
img_list = os.listdir(DATABASE_PATH)
np.random.shuffle(img_list)
for img_path in img_list:
    pathology = img_path.split("_")[0]
    print("True: ", pathology)
    img_path = os.path.join(DATABASE_PATH, img_path)
    # img_path = os.path.join(DATABASE_PATH, "viralpneumonia_DS5_person1424_virus_2437.png")
    img = cv2.imread(img_path, 0)
    x = np.expand_dims(img, axis=-1)
    x = preprocess(x)

    heatmap = grad_cam(model, x)
    overlayed_image = cv2.cvtColor(show_imgwithheat(img_path, heatmap), cv2.COLOR_BGR2RGB)
    cv2.imshow('', overlayed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("--------------------------------------------")