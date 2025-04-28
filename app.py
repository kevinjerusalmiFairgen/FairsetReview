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


# Function to load files
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
        st.error(f"Unsupported file type: {uploaded_file.name}")
        return None


# Function to process ZIP files
def extract_zip(zip_file):
    temp_dir = tempfile.mkdtemp()
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)
    return temp_dir


# Function to run fairset analysis for each extracted file
def run_fairset_analysis(priorfile, train, fairset, output_constraintsjson, output_structurejson, output_report_path):
    # Part 1: Extract prior file and make it JSON
    constraints_json, structure_json = priorFile_extract.priorFileExtract(priorfile)
    with open(output_constraintsjson, 'w') as f:
        f.write(json.dumps(constraints_json, indent=4))
    with open(output_structurejson, 'w') as f:
        f.write(json.dumps(structure_json, indent=4))

    # Part 2: Run Fairset check
    logic_instance = fairset_check.LogicFunctions("Dataset", train, fairset, empty_values=[])
    output_report = logic_instance.run_analysis(constraints_json)
    with open(output_report_path, 'w') as f:
        f.write(json.dumps(output_report, indent=4))

    # Part 3: Generate report
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
        priorfile_file = st.file_uploader("   ", type=["csv"])

    tab1, tab2 = st.tabs(["Fairset Review", "Structure & Prior File Extract"])

    with tab1:
        st.markdown("Upload train set, fairset, and prior file")

        if st.button("Run Analysis"):
            # Debugging output to check file availability
            st.write("Button clicked!")

            # Check if files are uploaded
            if train_file is None or fairset_file is None or priorfile_file is None:
                st.warning("Upload train, fairset, and prior file before running analysis!")
            else:
                st.write("Files uploaded successfully!")

                # Load the files
                train = load_file(train_file)
                priorfile = load_file(priorfile_file)

                # Handle ZIP for fairset
                if fairset_file.name.endswith(".zip"):
                    # Extract files from ZIP
                    temp_dir = extract_zip(fairset_file)
                    fairset_files = [f for f in os.listdir(temp_dir) if f.endswith(".sav")]
                    
                    if not fairset_files:
                        st.error("No .sav files found in the uploaded ZIP!")
                        return

                    st.write(f"Found {len(fairset_files)} .sav files in the ZIP.")
                    
                    # Run analysis for each fairset file
                    for fairset_filename in fairset_files:
                        fairset_path = os.path.join(temp_dir, fairset_filename)
                        fairset = load_file(fairset_path)

                        if fairset is not None:
                            st.write(f"Running analysis for: {fairset_filename}")

                            # Check columns are all right
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

                            # Run fairset analysis
                            df = run_fairset_analysis(**config)

                            st.write(f"### Results for {fairset_filename}")
                            st.dataframe(df, width=1000)

                            with open("outputs/template.xlsx", "rb") as file:
                                file_bytes = file.read()

                            st.download_button(
                                label="Download File",
                                data=file_bytes,
                                file_name="FairsetReview.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # Correct MIME type for Excel
                            )
                else:
                    # If the file is not a ZIP, just load it as usual
                    fairset = load_file(fairset_file)

                    if fairset is not None:
                        # Check columns are all right
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

                        # Run fairset analysis
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
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"  # Correct MIME type for Excel
                        )

    with tab2:
        st.markdown("Upload Prior file and get your Structure JSON")

        if st.button("Get JSONs"):
            # Debugging output to check file availability
            st.write("Get JSON button clicked!")

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


try:
    main()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.text(traceback.format_exc())
