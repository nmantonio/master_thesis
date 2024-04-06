import pandas as pd
from paths import *

import argparse
parser = argparse.ArgumentParser(description="Scrip to launch fold_train_and_val.py from excel sheet")
parser.add_argument('--device', type=str, required=True, help='GPU device to use during training and validation')
args = parser.parse_args()
device = args.device

df = pd.read_excel(TRAIN_SHEET, sheet_name="trains")
df["train_name"] = df["train_name"].astype(str)
print(df)
found = False
for idx, row in df.iterrows():
    if row["done"] == "no":
        found=True
        train_data = dict(row)
        del train_data["train_name"]
        del train_data["done"]
        
        train_name = f"{idx}_"        
        for key, value in train_data.items():
            train_name += f"{key}_{value}_"
        train_name = train_name[:-1]
        
        df.loc[idx, "done"] = "inprogress"
        df.loc[idx, "train_name"] = train_name
        train_data["train_name"] = train_name  
        train_data["device"] = device

      
        break

with pd.ExcelWriter(TRAIN_SHEET, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name="trains", index=False)
# df.to_excel(TRAIN_SHEET, index=False, header=True, sheet_name="trains")
del df
# print(dict(row))
# print(df)
from pprint import pprint
if found:
    pprint(train_data, width=1)
    
    instruction = "python fold_train_and_val.py"
    for arg, value in train_data.items():
        instruction += f' --{arg} {value}'
        
    print(instruction)
    # os.system(instruction)

import json
with open(os.path.join(TRAIN_PATH, train_name, "avg_results.json"), "r") as json_file:
    train_results = json.load(json_file)
train_results["train_name"] = train_name
results_df = pd.read_excel(RESULTS_SHEET)
results_df.loc[idx] = train_results
# results_df = pd.concat([results_df, pd.DataFrame(data=train_results, index=[0])])
results_df.to_excel(RESULTS_SHEET, index=False, header=True)
del results_df

df = pd.read_excel(TRAIN_SHEET, sheet_name="trains")
df.loc[idx, "done"] = "yes"
with pd.ExcelWriter(TRAIN_SHEET, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
    df.to_excel(writer, sheet_name="trains", index=False)
# df.to_excel(TRAIN_SHEET, index=False, header=True, sheet_name="trains")
del df