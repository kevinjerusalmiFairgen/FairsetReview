import scripts.priorFile_extract as priorFile_extract, scripts.fairset_check as fairset_check
import pandas as pd
import json
import scripts.generate_report as generate_report
import streamlit as st
import os
import tempfile
import pyreadstat
import traceback
import zipfile


def load_file(uploaded_file):            
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".sav"):
        temp_path = "temp.sav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        df, meta = pyreadstat.read_sav(temp_path)
        os.remove(temp_path)
        return df
    else:
        st.error(f"Unsupported file type: {uploaded_file.name}")
        return None


def load_file_from_path(filepath):
    try:
        if os.path.exists(filepath):
            print(f"Reading file from path: {filepath}")
            df, meta = pyreadstat.read_sav(filepath)
            return df
        else:
            print(f"Error: The file does not exist at {filepath}")
            return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def extract_files_from_zip(uploaded_file):
    extracted_files = []
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            for root, dirs, files in os.walk(temp_dir):
                for filename in files:
                    if filename.startswith("._"):
                        continue  # Skip MacOS junk
                    filepath = os.path.join(root, filename)
                    if "__MACOSX" in filepath:
                        continue  # Skip __MACOSX folder
                    if filename.endswith(('.csv', '.xlsx', '.sav')):
                        extracted_files.append(filepath)
                        
                        # Debugging line to print extracted file paths
                        print(f"Extracted file: {filepath}")  # Check that file is really extracted
        return extracted_files


def run_fairset_analysis(priorfile, train, fairset, output_constraintsjson, output_structurejson, output_report_path):
    ## <======= PART 1: Extract prior file and make it JSON =======>
    constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
    with open(output_constraintsjson, 'w') as f:
        f.write(json.dumps(constraints_json, indent=4))
    with open(output_structurejson, 'w') as f:
        f.write(json.dumps(structure_json, indent=4))

    ## <======= PART 2: Run Fairset check =======>
    
    # Ensure fairset is not None
    if fairset is None:
        raise ValueError("Fairset data is missing. Please ensure the fairset file is correctly loaded.")
    
    constraints = constraints_json
    # empty_values = file.get("empty_values", [])  # Add empty values possibility

    try:
        logic_instance = fairset_check.LogicFunctions("Dataset", train, fairset, empty_values=[])
        output_report = logic_instance.run_analysis(constraints)
        with open(output_report_path, 'w') as f:
            f.write(json.dumps(output_report, indent=4))
    except Exception as e:
        raise ValueError(f"Error running fairset analysis: {e}")
    
    ## <======= PART 3: Generate report =======>
    path = output_report_path
    df = generate_report.readOuput(path)
    generate_report.export_to_excel(df, "outputs/template.xlsx")

    return df



def main():
    st.title("Fairset Review Platform")

    with st.sidebar:
        st.markdown("## Upload Train Set")
        train_file = st.file_uploader(" ", type=["csv", "xlsx", "sav"])
        st.markdown("## Upload Fairset")
        fairset_file = st.file_uploader("  ", type=["csv", "xlsx", "sav", "zip"])
        st.markdown("## Upload Prior file")
        priorfile_file = st.file_uploader("   ", type=["csv"])

    tab1, tab2 = st.tabs(["Fairset Review", "Structure & Prior File Extract"]) 

    with tab1: 
        st.markdown("Upload train set, fairset and prior file")

        if st.button("Run Analysis"):
            if train_file is None or fairset_file is None or priorfile_file is None:
                st.warning("Upload train, fairset and prior file before running analysis!")
            if train_file is not None and fairset_file is not None and priorfile_file is not None:
                train = load_file(train_file)
                priorfile = load_file(priorfile_file)

                # Check columns are all right
                unknown_columns = priorFile_extract.check_columns_presence(priorfile, train, ["Source", "Target"])
                if unknown_columns:
                    bullet_list = "\n".join([f"- {col}" for col in unknown_columns])
                    st.error(f"The following column(s) from the Prior File are missing in the Data:\n{bullet_list}")
                    st.stop()

                if fairset_file.name.endswith(".zip"):
                    extracted_files = extract_files_from_zip(fairset_file)
                    if not extracted_files:
                        st.error("No usable file found inside the zip!")
                        st.stop()

                    for filepath in extracted_files:
                        fairset = load_file_from_path(filepath)
                        st.write(f"### Running analysis for: `{os.path.basename(filepath)}`")

                        config = {
                            "priorfile": priorfile,
                            "train": train,
                            "fairset": fairset,
                            "output_constraintsjson": "outputs/constraints.json",
                            "output_structurejson": "outputs/structure.json",
                            "output_report_path": "outputs/complete_report.json"
                        }

                        df = run_fairset_analysis(**config)

                        st.dataframe(df, width=1000)

                        with open("outputs/template.xlsx", "rb") as file:
                            file_bytes = file.read()

                        st.download_button(
                            label=f"Download File for {os.path.basename(filepath)}",
                            data=file_bytes,
                            file_name=f"FairsetReview_{os.path.basename(filepath)}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                else:
                    fairset = load_file(fairset_file)

                    config = {
                        "priorfile": priorfile,
                        "train": train,
                        "fairset": fairset,
                        "output_constraintsjson": "outputs/constraints.json",
                        "output_structurejson": "outputs/structure.json",
                        "output_report_path": "outputs/complete_report.json"
                    }

                    df = run_fairset_analysis(**config)

                    st.write("### ")
                    st.markdown(f"<h4>Fairset Review:", unsafe_allow_html=True)
                    st.dataframe(df, width=1000)

                    with open("outputs/template.xlsx", "rb") as file:
                        file_bytes = file.read()

                    st.download_button(
                        label="Download File",
                        data=file_bytes,
                        file_name="FairsetReview.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

    with tab2:
        st.markdown("Upload Prior file and get your Structure JSON")

        if st.button("Get JSONs"):
            if priorfile_file is None:
                st.warning("Upload a prior file and dataset before running analysis!")
            if priorfile_file is not None and train_file is not None:
                priorfile = load_file(priorfile_file)
                train = load_file(train_file)

                constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
                unknown_columns = priorFile_extract.check_columns_presence(priorfile, train, ["Source", "Target"])
                if unknown_columns:
                    bullet_list = "\n".join([f"- {col}" for col in unknown_columns])
                    st.error(f"The following column(s) from the Prior File are missing in the Data:\n{bullet_list}")
                    st.stop()

                with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as structure_tmp:
                    json.dump(structure_json, structure_tmp, indent=4)
                    structure_tmp_path = structure_tmp.name

                with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as constraints_tmp:
                    json.dump(constraints_json, constraints_tmp, indent=4)
                    constraints_tmp_path = constraints_tmp.name

                with open(structure_tmp_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Structure JSON",
                        data=f,
                        file_name="structure.json",
                        mime="application/json"
                    )

                with open(constraints_tmp_path, "rb") as f:
                    st.download_button(
                        label="⬇️ Download Constraints JSON",
                        data=f,
                        file_name="constraints.json",
                        mime="application/json"
                    )


try:
    main()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.text(traceback.format_exc())
