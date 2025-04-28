import scripts.priorFile_extract as priorFile_extract
import scripts.fairset_check as fairset_check
import scripts.generate_report as generate_report
import pandas as pd
import json
import streamlit as st
import os
import tempfile
import pyreadstat
import zipfile
import io
import traceback

# Make sure the outputs folder exists
os.makedirs("outputs", exist_ok=True)

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
    else:
        raise ValueError(f"Unsupported file type: {uploaded_file.name}")

def load_wrapper(uploaded_file):
    dfs = {}
    if uploaded_file.name.endswith(".zip"):
        with zipfile.ZipFile(uploaded_file) as z:
            for file_info in z.infolist():
                if file_info.is_dir():
                    continue
                # Skip MACOSX hidden files and files starting with ._
                if "__MACOSX" in file_info.filename or os.path.basename(file_info.filename).startswith("._"):
                    continue
                with z.open(file_info) as extracted_file:
                    extracted_file = io.BytesIO(extracted_file.read())
                    extracted_file.name = file_info.filename
                    try:
                        dfs[file_info.filename] = load_file(extracted_file)
                    except Exception:
                        st.warning(f"Skipping unsupported or corrupted file: {file_info.filename}")
        return dfs
    else:
        try:
            df = load_file(uploaded_file)
            return {uploaded_file.name: df} if df is not None else {}
        except Exception:
            st.warning(f"Skipping unsupported or corrupted file: {uploaded_file.name}")
            return {}

def run_fairset_analysis(priorfile, train, fairset, output_constraintsjson, output_structurejson, output_report_path):
    constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
    with open(output_constraintsjson, 'w') as f:
        f.write(json.dumps(constraints_json, indent=4))
    with open(output_structurejson, 'w') as f:
        f.write(json.dumps(structure_json, indent=4))

    constraints = constraints_json
    logic_instance = fairset_check.LogicFunctions("Dataset", train, fairset, empty_values=[])
    output_report = logic_instance.run_analysis(constraints)
    with open(output_report_path, 'w') as f:
        f.write(json.dumps(output_report, indent=4))

    df = generate_report.readOuput(output_report_path)
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
                st.stop()

            train = load_file(train_file)
            priorfile = load_file(priorfile_file)

            unknown_columns = priorFile_extract.check_columns_presence(priorfile, train, ["Source", "Target"])
            if unknown_columns:
                bullet_list = "\n".join([f"- {col}" for col in unknown_columns])
                st.error(f"The following column(s) from the Prior File are missing in the Data:\n{bullet_list}")
                st.stop()

            fairsets = load_wrapper(fairset_file)

            if not fairsets:
                st.error("No valid fairset datasets found to process!")
                st.stop()

            st.info(f"Found {len(fairsets)} valid fairset(s).")

            # 🛠 Store processed results
            processed_fairsets = []

            for filename, fairset in fairsets.items():
                if fairset is None:
                    continue

                st.write(f"### Running analysis for `{filename}`...")

                safe_filename = filename.replace("/", "_").replace("\\", "_").replace(".csv", "").replace(".xlsx", "").replace(".sav", "")
                output_constraintsjson = f"outputs/constraints_{safe_filename}.json"
                output_structurejson = f"outputs/structure_{safe_filename}.json"
                output_report_path = f"outputs/complete_report_{safe_filename}.json"
                output_excel_path = f"outputs/FairsetReview_{safe_filename}.xlsx"

                config = {
                    "priorfile": priorfile,
                    "train": train,
                    "fairset": fairset,
                    "output_constraintsjson": output_constraintsjson,
                    "output_structurejson": output_structurejson,
                    "output_report_path": output_report_path
                }

                try:
                    df = run_fairset_analysis(**config)
                    generate_report.export_to_excel(df, output_excel_path)

                    # ✨ Save results for display later
                    processed_fairsets.append({
                        "filename": filename,
                        "safe_filename": safe_filename,
                        "df": df,
                        "output_excel_path": output_excel_path
                    })

                except Exception as e:
                    st.error(f"Failed on {filename}: {str(e)}")
                    st.text(traceback.format_exc())

            # 🖥️ After all fairsets are processed, show results
            for item in processed_fairsets:
                st.markdown(f"<h4>Results for {item['filename']}:</h4>", unsafe_allow_html=True)
                st.dataframe(item['df'], width=1000)

                with open(item['output_excel_path'], "rb") as file:
                    file_bytes = file.read()

                st.download_button(
                    label=f"⬇️ Download Report for {item['filename']}",
                    data=file_bytes,
                    file_name=f"FairsetReview_{item['safe_filename']}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

    with tab2:
        st.markdown("Upload Prior file and get your Structure JSON")

        if st.button("Get JSONs"):
            if priorfile_file is None or train_file is None:
                st.warning("Upload a prior file and dataset before running analysis!")
                st.stop()

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
