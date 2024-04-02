import pickle

def load_encoder(encoder_path):
    with open(encoder_path, "rb") as encoder_file: 
        encoder = pickle.load(encoder_file)
        
    return encoder