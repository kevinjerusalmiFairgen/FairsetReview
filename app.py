import scripts.priorFile_extract as priorFile_extract
import scripts.fairset_check as fairset_check
import pandas as pd
import json
import scripts.generate_report as generate_report
import streamlit as st
import os
import tempfile
import pyreadstat
import traceback
import zipfile
import io


def load_file(uploaded_file):            
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        return pd.read_excel(uploaded_file)
    elif uploaded_file.name.endswith(".sav"):
        temp_path = "temp.sav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        df = pd.read_spss(temp_path)
        os.remove(temp_path)
        return df
    elif uploaded_file.name.endswith(".zip"):
        return extract_zip(uploaded_file)
    else:
        st.error(f"Unsupported file type: {uploaded_file.name}")
        return None


def extract_zip(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    extracted_files = {}
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if file.lower().endswith(".csv"):
                df = pd.read_csv(filepath)
            elif file.lower().endswith(".xlsx"):
                df = pd.read_excel(filepath)
            elif file.lower().endswith(".sav"):
                df = pd.read_spss(filepath)
            else:
                continue

            if "train" in file.lower():
                extracted_files["train"] = df
            elif "fairset" in file.lower():
                extracted_files["fairset"] = df
            elif "prior" in file.lower():
                extracted_files["priorfile"] = df

    return extracted_files


def run_fairset_analysis(priorfile, train, fairset, output_constraintsjson, output_structurejson, output_report_path):
    ## <======= PART 1: Extract prior file and make it JSON =======>
    constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
    with open(output_constraintsjson, 'w') as f:
        f.write(json.dumps(constraints_json, indent=4))
    with open(output_structurejson, 'w') as f:
        f.write(json.dumps(structure_json, indent=4))

    ## <======= PART 2: Run Fairset check =======>
    constraints = constraints_json

    logic_instance = fairset_check.LogicFunctions("Dataset", train, fairset, empty_values=[])
    output_report = logic_instance.run_analysis(constraints)
    with open(output_report_path, 'w') as f:
        f.write(json.dumps(output_report, indent=4))

    ## <======= PART 3: Generate report =======>
    path = output_report_path
    df = generate_report.readOuput(path)
    generate_report.export_to_excel(df, "outputs/template.xlsx")

    return df


def main():
    st.title("Fairset Review Platform")

    with st.sidebar:
        st.markdown("## Upload Files")
        uploaded_files = st.file_uploader(
            "Upload Train, Fairset, Prior file OR ZIP",
            type=["csv", "xlsx", "sav", "zip"],
            accept_multiple_files=False
        )

    tab1, tab2 = st.tabs(["Fairset Review", "Structure & Prior File Extract"]) 

    if uploaded_files is None:
        st.warning("Please upload a file first.")
        return

    data = load_file(uploaded_files)

    if isinstance(data, dict):
        train = data.get("train")
        fairset = data.get("fairset")
        priorfile = data.get("priorfile")
    else:
        # Single file upload handling if needed later
        st.error("Please upload a .zip containing train, fairset, and prior files.")
        return

    with tab1:
        st.markdown("Upload train set, fairset and prior file")

        if st.button("Run Analysis"):
            if train is None or fairset is None or priorfile is None:
                st.warning("Train, fairset, and prior file must be uploaded!")
                st.stop()

            unknown_columns = priorFile_extract.check_columns_presence(priorfile, train, ["Source", "Target"])
            if unknown_columns:
                bullet_list = "\n".join([f"- {col}" for col in unknown_columns])
                st.error(f"The following column(s) from the Prior File are missing in the Data:\n{bullet_list}")
                st.stop()

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
            st.markdown(f"<h4>Fairset Review:</h4>", unsafe_allow_html=True)
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
            if priorfile is None:
                st.warning("Upload a prior file inside zip!")
                st.stop()

            constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)

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

# --- Always keep try-except inside main ---
try:
    main()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.text(traceback.format_exc())
