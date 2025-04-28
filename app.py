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


def load_file(uploaded_file, expected_type):            
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
        return extract_from_zip(uploaded_file, expected_type)
    else:
        st.error(f"Unsupported file type: {uploaded_file.name}")
        return None


def extract_from_zip(zip_file, expected_type):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            filepath = os.path.join(root, file)
            if filepath.endswith(('.csv', '.xlsx', '.sav')):
                return read_file(filepath)

    st.error(f"No valid file (csv/xlsx/sav) found inside the ZIP.")
    return None



def read_file(filepath):
    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    elif filepath.endswith(".xlsx"):
        return pd.read_excel(filepath)
    elif filepath.endswith(".sav"):
        return pd.read_spss(filepath)
    else:
        return None


def run_fairset_analysis(priorfile, train, fairset, output_constraintsjson, output_structurejson, output_report_path):
    constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
    with open(output_constraintsjson, 'w') as f:
        f.write(json.dumps(constraints_json, indent=4))
    with open(output_structurejson, 'w') as f:
        f.write(json.dumps(structure_json, indent=4))

    logic_instance = fairset_check.LogicFunctions("Dataset", train, fairset, empty_values=[])
    output_report = logic_instance.run_analysis(constraints_json)
    with open(output_report_path, 'w') as f:
        f.write(json.dumps(output_report, indent=4))

    df = generate_report.readOuput(output_report_path)
    generate_report.export_to_excel(df, "outputs/template.xlsx")

    return df


def main():
    st.title("Fairset Review Platform")

    with st.sidebar:
        st.markdown("## Upload Train Set")
        train_file = st.file_uploader(" ", type=["csv", "xlsx", "sav", "zip"])
        st.markdown("## Upload Fairset")
        fairset_file = st.file_uploader("  ", type=["csv", "xlsx", "sav", "zip"])
        st.markdown("## Upload Prior file")
        priorfile_file = st.file_uploader("   ", type=["csv", "zip"])

    tab1, tab2 = st.tabs(["Fairset Review", "Structure & Prior File Extract"]) 

    with tab1:
        st.markdown("Upload train set, fairset and prior file")

        if st.button("Run Analysis"):
            if train_file is None or fairset_file is None or priorfile_file is None:
                st.warning("Upload train, fairset and prior file before running analysis!")
                st.stop()

            train = load_file(train_file, "train")
            fairset = load_file(fairset_file, "fairset")
            priorfile = load_file(priorfile_file, "priorfile")

            if train is None or fairset is None or priorfile is None:
                st.error("Problem reading one of the files. Check uploads.")
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
            if priorfile_file is None or train_file is None:
                st.warning("Upload prior and train files first!")
                st.stop()

            priorfile = load_file(priorfile_file, "priorfile")
            train = load_file(train_file, "train")

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
