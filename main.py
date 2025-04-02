import scripts.priorFile_extract as priorFile_extract, scripts.fairset_check as fairset_check
import pandas as pd
import json
import scripts.generate_report as generate_report


def run_fairset_analysis(priorfile_path, train_path, fairset_path, output_constraintsjson, output_structurejson, output_report_path):

    ## <======= PART 1: Extract prior file and make it JSON =======>
    constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile_path)
    with open(output_constraintsjson, 'w') as f:
        f.write(json.dumps(constraints_json, indent=4))
    with open(output_structurejson, 'w') as f:
        f.write(json.dumps(structure_json, indent=4))


    ## <======= PART 2: Run Fairset check =======>
    data = {
            "train_path": train_path,
            "fairset_path": fairset_path,
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
    output_report = logic_instance.run_analysis(constraints)
    with open(output_report_path, 'w') as f:
        f.write(json.dumps(output_report, indent=4))


    ## <======= PART 3: Generate report =======>
    path = output_report_path
    df = generate_report.readOuput(path)
    generate_report.export_to_excel(df, "outputs/template.xlsx")


config = {
    "priorfile_path": "data/priorfile.csv",
    "train_path": "Data/Leger_Royal Canadian Mint_data1.sav",
    "fairset_path": "Data/fairset-legerroyalcanada.sav",
    "output_constraintsjson": "outputs/constraints.json",
    "output_structurejson": "outputs/structure.json",
    "output_report_path": "outputs/complete_report.json"
}

run_fairset_analysis(**config)