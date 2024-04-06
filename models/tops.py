from keras.layers import Dense, GlobalAveragePooling2D


def get_top(top_idx):
    if int(top_idx) == 1: 
        return top_1
    elif int(top_idx) == 2:
        return top_2

def top_1(x):
    x = GlobalAveragePooling2D()(x)
    return x

def top_2(x):
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(256, activation='relu')(x)
    return x