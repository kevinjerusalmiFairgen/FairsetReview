import scripts.priorFile_extract as priorFile_extract
from scripts.fairset_check import LogicFunctions
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

                # Always take only the basename (ignore internal folders inside ZIP)
                clean_name = os.path.basename(file_info.filename)

                with z.open(file_info) as extracted_file:
                    extracted_file = io.BytesIO(extracted_file.read())
                    extracted_file.name = clean_name  # <-- assign clean name for load_file

                    try:
                        dfs[clean_name] = load_file(extracted_file)
                    except Exception:
                        st.warning(f"Skipping unsupported or corrupted file: {clean_name}")
        return dfs
    else:
        try:
            df = load_file(uploaded_file)
            return {uploaded_file.name: df} if df is not None else {}
        except Exception:
            st.warning(f"Skipping unsupported or corrupted file: {uploaded_file.name}")
            return {}


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

            try:
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

                processing_results = []  # <--- collect results here

                temp_dir = tempfile.mkdtemp()

                # Silent processing first
                for filename, fairset in fairsets.items():
                    if fairset is None:
                        continue

                    safe_filename = filename.replace("/", "_").replace("\\", "_").replace(".csv", "").replace(".xlsx", "").replace(".sav", "")
                    output_constraintsjson = os.path.join(temp_dir, f"constraints_{safe_filename}.json")
                    output_structurejson = os.path.join(temp_dir, f"structure_{safe_filename}.json")
                    output_report_path = os.path.join(temp_dir, f"complete_report_{safe_filename}.json")
                    output_excel_path = os.path.join(temp_dir, f"FairsetReview_{safe_filename}.xlsx")

                    config = {
                        "priorfile": priorfile,
                        "train": train,
                        "fairset": fairset,
                        "output_constraintsjson": output_constraintsjson,
                        "output_structurejson": output_structurejson,
                        "output_report_path": output_report_path
                    }

                    try:
                        df = LogicFunctions.run_analysis(**config)
                        generate_report.export_to_excel(df, output_excel_path)

                        processing_results.append({
                            "filename": filename,
                            "safe_filename": safe_filename,
                            "df": df,
                            "excel_path": output_excel_path
                        })

                    except Exception as e:
                        processing_results.append({
                            "filename": filename,
                            "error": str(e)
                        })

                # Now AFTER everything, show results
                if processing_results:
                    st.success(f"✅ Processed {len(processing_results)} fairsets!")

                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zip_file:
                        for item in processing_results:
                            if "excel_path" in item:
                                with open(item['excel_path'], "rb") as f:
                                    file_data = f.read()
                                    zip_file.writestr(f"FairsetReview_{item['safe_filename']}.xlsx", file_data)
                    zip_buffer.seek(0)

                    # Display all results
                    for item in processing_results:
                        if "error" in item:
                            st.error(f"❌ {item['filename']} failed: {item['error']}")
                        else:
                            st.markdown(f"### Results for `{item['filename']}`")
                            st.dataframe(item["df"], use_container_width=True)

                    # Single download for all
                    st.download_button(
                        label="⬇️ Download ALL Reports (.zip)",
                        data=zip_buffer,
                        file_name="FairsetReports.zip",
                        mime="application/zip"
                    )

                else:
                    st.warning("⚠️ No fairsets processed successfully.")

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")
                st.text(traceback.format_exc())

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
