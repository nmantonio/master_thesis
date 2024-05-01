from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import pickle
from paths import *

os.makedirs(ENCODERS_PATH, exist_ok=True)

df = pd.read_csv(CLASSIFICATION_CSV)
classification = pd.DataFrame(df["classification"])

encoder = OneHotEncoder()
encoder.fit(classification)
print("--------- CLASSIFICATION ---------")
print("IN: ", encoder.feature_names_in_)
print("OUT: ", encoder.get_feature_names_out())

with open(CLASSIFICATION_ENCODER, "wb") as f: 
    pickle.dump(encoder, f)
    
del encoder

print()
# ------------------------------------------------------
df = pd.read_csv(CLASSIFICATION_CSV)
df = df[df["classification"] != "normal"]
classification = pd.DataFrame(df["classification"])
encoder = OneHotEncoder()
encoder.fit(classification)
print("--------- ABNORMAL CLASSIFICATION ---------")
print("IN: ", encoder.feature_names_in_)
print("OUT: ", encoder.get_feature_names_out())

with open(ABNORMAL_CLASSIFICATION_ENCODER, "wb") as f: 
    pickle.dump(encoder, f)
    
del encoder

print()
# ------------------------------------------------------
df = pd.read_csv(DETECTION_CSV)
detection = pd.DataFrame(df["detection"])
encoder = OneHotEncoder()
encoder.fit(detection)
print("--------- DETECTION ---------")
print("IN: ", encoder.feature_names_in_)
print("OUT: ", encoder.get_feature_names_out())

with open(DETECTION_ENCODER, "wb") as f: 
    pickle.dump(encoder, f)
    
del encoder