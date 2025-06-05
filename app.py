import sys
import os
import scripts.priorFile_extract as priorFile_extract, scripts.fairset_check as fairset_check, scripts.scrapper as scrapper
import pandas as pd
import json
import scripts.generate_report as generate_report
import streamlit as st
import tempfile
import pyreadstat
import traceback


APP_ROOT = os.path.dirname(os.path.abspath(__file__))
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)


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
    
    st.title("Pilot Manager")

    with st.sidebar:
            st.markdown("## Upload Train Set")
            train_file = st.file_uploader(" ", type=["csv", "xlsx", "sav"])
            st.markdown("## Upload Fairset")
            fairset_file = st.file_uploader("  ", type=["csv", "xlsx", "sav"])
            st.markdown("## Upload Prior file")
            priorfile_file = st.file_uploader("   ", type=["csv"])

    tab1, tab2, tab3 = st.tabs(["Fairset Review", "Structure & Logics Extract", "Boost Results"]) 
    #tab1, tab2 = st.tabs(["Fairset Review", "Structure & Logics Extract"]) 


    with tab1: 
        st.markdown("Upload train set, fairset and prior file")

        if st.button("Run Analysis"):
            if train_file is None or fairset_file is None or priorfile_file is None:
                st.warning("Upload train, fairset and prior file before running analysis!")
            if train_file is not None and fairset_file is not None and priorfile_file is not None:
                train = load_file(train_file)
                fairset = load_file(fairset_file)
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

        if st.button("Get JSONs"):
            if priorfile_file is None:
                st.warning("Upload a prior file and dataset before running analysis!")
            if priorfile_file is not None and train_file is not None:
                priorfile = load_file(priorfile_file)
                train = load_file(train_file)

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
    
    with tab3:
        # Initialize data
        environements = ["Classic Environnement", "Kevin's Environnement"]
        emails = ["inspector@fairgen.ai", "inspector+private@fairgen.ai"]

        # Initialize session state
        if "selected_email" not in st.session_state:
            st.session_state.selected_email = emails[0]
        if 'df_scraped' not in st.session_state:
            st.session_state.df_scraped = None

        # Dropdown for environments
        col1, _ = st.columns([1, 2])
        with col1:
            selected_env = st.selectbox("Select an environment", environements, index=0)

        # Update selected_email based on environment
        if selected_env == environements[0]:
            st.session_state.selected_email = emails[0]
        else:
            st.session_state.selected_email = emails[1]

        # Input field
        project_url = st.text_input("# Fetch parallel Tests results from URL")

        # Run scraper
        if st.button("Run Scraper") and project_url:
            scrapper.scrap_boostresults(project_url)

            if st.session_state.df_scraped is not None:
                df = st.session_state.df_scraped
                df["Boost MAE (%)"] = pd.to_numeric(df["Boost MAE (%)"], errors='coerce')
                df["Training MAE (%)"] = pd.to_numeric(df["Training MAE (%)"], errors='coerce')
                df["Niche Size"] = pd.to_numeric(df["Niche Size"], errors='coerce')
                st.session_state.df_scraped = df  # update cleaned df

        # If scraped data is available
        if st.session_state.df_scraped is not None:
            df = st.session_state.df_scraped

            if df.shape[0] > 2:
                st.markdown("### Filter Metrics by Niche Size")

                min_val = int(df["Niche Size"].min())
                max_val = int(df["Niche Size"].max())

                min_size, max_size = st.slider(
                    "Select Niche Size Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    step=1,
                    key="niche_slider"
                )

                filtered = df[(df["Niche Size"] >= min_size) & (df["Niche Size"] <= max_size)]

                if not filtered.empty:
                    st.markdown("### Filtered Scraped Data")
                    st.dataframe(filtered)

                    with st.container():
                        st.markdown(
                            """
                            <div style="background-color: #f0f0f0; padding: 20px; border-radius: 10px;">
                            """,
                            unsafe_allow_html=True
                        )

                        boost_win_rate = (filtered["Boost MAE (%)"] < filtered["Training MAE (%)"]).mean() * 100
                        avg_added_value = ((filtered["Training MAE (%)"] - filtered["Boost MAE (%)"]) / filtered["Training MAE (%)"]).mean() * 100

                        st.markdown(f"**Boost Win Rate:** {boost_win_rate:.2f}%")
                        st.markdown(f"**Average Added Value:** {avg_added_value:.2f}%")

                        st.markdown("</div>", unsafe_allow_html=True)
                else:
                    st.warning("No data in selected range.")
            else:
                st.markdown("### Full Scraped Data")
                st.dataframe(df)

        # Optional reset
        if st.button("Clear All"):
            st.session_state.df_scraped = None
            st.rerun()

try:
    main()
except Exception as e:
    st.error(f"An error occurred: {e}")
    st.text(traceback.format_exc())
