import scripts.priorFile_extract as priorFile_extract, scripts.fairset_check as fairset_check
import pandas as pd
import json
import scripts.generate_report as generate_report


## <======= PART 1: Extract prior file and make it JSON =======>
df = priorFile_extract.extractAndPreprocessing("1bbad92c6e1081309253c9729bd9232b", index_order=None)
constraints_json, structure_json = priorFile_extract.createJSONs(df)
with open('outputs/constraints.json', 'w') as f:
    f.write(json.dumps(constraints_json, indent=4))
with open('outputs/structure.json', 'w') as f:
    f.write(json.dumps(structure_json, indent=4))


## <======= PART 2: Run Fairset check =======>
data = {
        "train_path": "Data/Leger_Royal Canadian Mint_data1.sav",
        "fairset_path": "Data/fairset-legerroyalcanada.sav",
        #"empty_values": ["nan", "NO TO:", ""]
}

constraints =  constraints_json
# empty_values = file.get("empty_values", [])  # Add empty values possibility

if data["train_path"].endswith(".sav"):
    train = pd.read_spss(data["train_path"])
    fairset = pd.read_spss(data["fairset_path"])
elif data["train_path"].endswith(".xlsx"):
    train = pd.read_excel(data["train_path"])
    fairset = pd.read_excel(data["fairset_path"])
elif data["train_path"].endswith(".csv"):
    train = pd.read_csv(data["train_path"])
    fairset = pd.read_csv(data["fairset_path"])

logic_instance = fairset_check.LogicFunctions("Dataset", train, fairset, empty_values=[])
output = logic_instance.run_analysis(constraints)
with open('outputs/output.json', 'w') as f:
    f.write(json.dumps(output, indent=4))

## <======= PART 3: Generate report =======>
path = "outputs/output.json"
df = generate_report.readOuput(path)
generate_report.export_to_excel(df, "outputs/template.xlsx")
