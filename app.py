import sys
import os
import scripts.priorFile_extract as priorFile_extract, scripts.fairset_check as fairset_check
import pandas as pd
import json
import scripts.generate_report as generate_report
import streamlit as st
import tempfile
import pyreadstat
import traceback
from datetime import datetime
import numpy as np



APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)


def to_jsonable(obj):
    # NumPy & pandas scalars
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Arrays / Series / Index
    if isinstance(obj, (np.ndarray, pd.Series, pd.Index)):
        return obj.tolist()

    # Timestamps / datetimes
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()

    # Sets/tuples
    if isinstance(obj, (set, tuple)):
        return list(obj)

    # NaN/NaT handling (optional)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass

    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def load_file(uploaded_file, japanase_chars=False, convert_categoricals = True):
                    if uploaded_file.name.endswith(".csv"):
                        if japanase_chars:
                            return pd.read_csv(uploaded_file, encoding='shift_jis')
                        return pd.read_csv(uploaded_file)
                    elif uploaded_file.name.endswith(".xlsx"):
                        return pd.read_excel(uploaded_file)
                    elif uploaded_file.name.endswith(".sav"):
                        temp_path = "temp.sav"
                        with open(temp_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                        df = pd.read_spss(temp_path)#, convert_categoricals=convert_categoricals)
                        os.remove(temp_path)
                        
                        return df
                    else:
                        st.error(f"Unsupported file type: {uploaded_file.name}")
                        return None


def run_fairset_analysis(priorfile, train, fairset, output_constraintsjson, output_structurejson, output_report_path):

    ## <======= PART 1: Extract prior file and make it JSON =======>
    constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
    with open(output_constraintsjson, 'w') as f:
        f.write(json.dumps(constraints_json, indent=4))
    with open(output_structurejson, 'w') as f:
        f.write(json.dumps(structure_json, indent=4))


    ## <======= PART 2: Run Fairset check =======>
    

    constraints =  constraints_json
    # empty_values = file.get("empty_values", [])  # Add empty values possibility

    logic_instance = fairset_check.LogicFunctions("Dataset", train, fairset, empty_values=[])
    output_report = logic_instance.run_analysis(constraints)
    with open(output_report_path, 'w', encoding='utf-8') as f:
        st.write(output_report)
        f.write(json.dumps(output_report, indent=4, ensure_ascii=False, default=to_jsonable))


    ## <======= PART 3: Generate report =======>
    path = output_report_path
    df = generate_report.readOuput(path)
    generate_report.export_to_excel(df, "outputs/template.xlsx", fairset_len=fairset.shape[0])

    return df


def main():
    if "init" not in st.session_state:
        st.session_state.init = True
        # optionally add a delay to ensure full sync
        import time
        time.sleep(0.5)

    st.write("Session initialized.")
    st.title("Pilot Manager")

    with st.sidebar:
            st.markdown("## Upload Train Set")
            train_file = st.file_uploader(" ", type=["csv", "xlsx", "sav"])
            st.markdown("## Upload Fairset")
            fairset_file = st.file_uploader("  ", type=["csv", "xlsx", "sav"])
            st.markdown("## Upload Prior file")
            priorfile_file = st.file_uploader("   ", type=["csv"])

    tab1, tab2 = st.tabs(["Fairset Review", "Structure & Logics Extract"]) #, "Boost Results"]) 
    #tab1, tab2 = st.tabs(["Fairset Review", "Structure & Logics Extract"]) 
    japanase_chars = False


    with tab1: 
        st.markdown("Upload train set, fairset and prior file")
        japanase_chars = st.checkbox("Japanese survey")
        japanase_chars_fairset = st.checkbox("Japanese fairset")

        if st.button("Run Analysis"):
            if train_file is None or fairset_file is None or priorfile_file is None:
                st.warning("Upload train, fairset and prior file before running analysis!")
            if train_file is not None and fairset_file is not None and priorfile_file is not None:
                train = load_file(train_file, japanase_chars)
                fairset = load_file(fairset_file, japanase_chars_fairset)
                priorfile = load_file(priorfile_file)

                # Check columns are all right
                unknown_columns = priorFile_extract.check_columns_presence(priorfile, train, ["Source", "Target"])
                if unknown_columns and unknown_columns!= [""]:
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
                st.markdown(f"<h4>Fairset Review:", unsafe_allow_html=True)
                st.dataframe(df, width=1000)

                with open("outputs/template.xlsx", "rb") as file:
                    file_bytes = file.read()

                st.download_button(
                    label="Download File",
                    data=file_bytes,
                    file_name="FairsetReview.xlsx",
                    mime="text/csv"  # Adjust MIME type depending on your file
                )

    with tab2:

        st.markdown("Upload Prior file and get your Structure JSON")
        japanase_chars = st.checkbox("Japanese survey characters")


        if st.button("Get JSONs"):
            if priorfile_file is None:
                st.warning("Upload a prior file and dataset before running analysis!")
            if priorfile_file is not None and train_file is not None:
                priorfile = load_file(priorfile_file)
                train = load_file(train_file, japanase_chars)

                constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
                unknown_columns = priorFile_extract.check_columns_presence(priorfile, train, ["Source", "Target"])
                if unknown_columns and unknown_columns!= [""]:
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
