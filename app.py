import scripts.priorFile_extract as priorFile_extract, scripts.fairset_check as fairset_check
import pandas as pd
import json
import scripts.generate_report as generate_report
import streamlit as st
import os
import tempfile
import pyreadstat


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
    with open(output_report_path, 'w') as f:
        f.write(json.dumps(output_report, indent=4))


    ## <======= PART 3: Generate report =======>
    path = output_report_path
    df = generate_report.readOuput(path)
    generate_report.export_to_excel(df, "outputs/template.xlsx")

    return df


def main():
    
    st.title("Fairset Review Platform")

    st.markdown("Upload train set, fairset and prior file")

    with st.sidebar:
        st.markdown("## Upload Train Set")
        train_file = st.file_uploader(" ", type=["csv", "xlsx", "sav"])
        st.markdown("## Upload Fairset")
        fairset_file = st.file_uploader("  ", type=["csv", "xlsx", "sav"])
        st.markdown("## Upload Prior file")
        priorfile_file = st.file_uploader("   ", type=["csv"])

    col1, _, col2, _ = st.columns(4)

    with col1:
        if st.button("Run Analysis"):
            if train_file is None or fairset_file is None or priorfile_file is None:
                st.warning("Upload train, fairset and prior file before running analysis!")
            if train_file is not None and fairset_file is not None and priorfile_file is not None:
                train = load_file(train_file)
                fairset = load_file(fairset_file)
                priorfile = load_file(priorfile_file)

                # Check columns are all right
                unknown_columns = priorFile_extract. check_columns_presence(priorfile, train, ["Source", "Target"])
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

                st.dataframe(df, width=1000)

                with open("outputs/template.xlsx", "rb") as file:
                    file_bytes = file.read()

                st.download_button(
                    label="Download File",
                    data=file_bytes,
                    file_name="FairsetReview.csv",
                    mime="text/csv"  # Adjust MIME type depending on your file
                )
    with col2:
        if st.button("Get JSONs"):
            priorfile = load_file(priorfile_file)
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


# try:
#     main()
# except Exception as e:
#     st.error(f"An error occurred: {e}")
main()
