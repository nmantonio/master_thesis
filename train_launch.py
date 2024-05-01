import pandas as pd
from paths import *

import argparse
parser = argparse.ArgumentParser(description="Scrip to launch fold_train_and_val.py from excel sheet")
parser.add_argument('--device', type=str, required=True, help='GPU device to use during training and validation')
args = parser.parse_args()
device = args.device


while True: 
    df = pd.read_excel(TRAIN_SHEET, sheet_name="trains")
    df["train_name"] = df["train_name"].astype(str)

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
            
            df.loc[idx, "done"] = f"{device}_inprogress"
            df.loc[idx, "train_name"] = train_name
            train_data["train_name"] = train_name  
            train_data["device"] = device

        
            break
    else: # If all trainings are done or in progress, end file.
        break

    with pd.ExcelWriter(TRAIN_SHEET, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="trains", index=False)
    # df.to_excel(TRAIN_SHEET, index=False, header=True, sheet_name="trains")
    del df

    # from pprint import pprint
    # pprint(train_data, width=1)
    
    instruction = "python fold_train_and_val.py"
    for arg, value in train_data.items():
        instruction += f' --{arg} {value}'
        
    print(instruction)
    os.system(instruction)

    import json
    with open(os.path.join(TRAIN_PATH, train_name, "avg_results.json"), "r") as json_file:
        train_results = json.load(json_file)
        
    train_info = dict(row)
    train_info.pop("patience", None)
    train_info.pop("batch_size", None)
    train_info.pop("done", None)
    train_info.pop("optimizer", None)
    train_info.pop("epochs", None)
    train_info.pop("lr", None)
    train_info.pop("train_name", None)
    
    train_info.update(train_results)
    train_info["train_name"] = train_name
    # train_results["train_name"] = train_name
    if not os.path.exists(RESULTS_SHEET):
    # If the Excel file doesn't exist, create it and write the DataFrame to it
        results_df = pd.DataFrame(data=train_info, index=[0])
        # results_df = results_df.T
        results_df.to_excel(RESULTS_SHEET, index=False, header=True)
    else:
        # If the Excel file exists, load it and append the new row
        results_df = pd.read_excel(RESULTS_SHEET)
        results_df = pd.concat([results_df, pd.DataFrame(data=train_info, index=[0])])
        results_df.to_excel(RESULTS_SHEET, index=False, header=True)
    del results_df

    df = pd.read_excel(TRAIN_SHEET, sheet_name="trains")
    df.loc[idx, "done"] = "yes"
    with pd.ExcelWriter(TRAIN_SHEET, engine='openpyxl', mode="a", if_sheet_exists="replace") as writer:
        df.to_excel(writer, sheet_name="trains", index=False)
    # df.to_excel(TRAIN_SHEET, index=False, header=True, sheet_name="trains")
    del df