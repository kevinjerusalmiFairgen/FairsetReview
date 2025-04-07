import json
import pandas as pd
import xlsxwriter
import math
import re
import streamlit as st


def convert_type(s):
    if s.startswith("Block"):
        parts = s.split(" ")
        if "Multi-to-Single" in parts:
            s = "Compound Skip Logic"
        elif "Multi-to-Multi" in parts:
            s = "Piping"
        else:
            s = "Skip Logic"
    elif s.startswith("Force"):
        s = "Mandatory Logic"
    elif s.startswith("Recoding"):
        s = "Recoding"

    translation = {
        "Block": "Skip Logic",
        "Force": "Mandatory Logic",
        "Compound Skip Logic": "Compound Skip Logic",
        "Piping": "Piping",
        "Sum": "Calculation Logic",
        "Recoding": "Recodes/Hidden Variables",
        "None of the Above": "Exclusive",  
        "All of the Above": "Exclusive", 
        "Count": "Selection Limit Control",
        "Uniqueness": "Ranking",
    }

    return translation.get(s, s)


def readOuput(path):
    with open(path, 'r') as file:
        data = json.load(file)
    data = [
        {
            **item,
            "Detail": item["Description"] if not item.get("Detail") else item["Detail"]
        }
        for item in data
        if item and item.get("is_valid") == False
    ]
        
    if data == [] or data.empty():
        st.write("Empty Data")

    if not isinstance(data, list):
        st.error("Expected data to be a list, got something else.")     

    
    st.write(pd.DataFrame(data).columns)
    df = pd.DataFrame(data)[["Type", "is_supported", "Dataframe", "Detail", "Percentage_of_valid_rows", "Rows"]]

    df["Logic Type"] = df["Type"].apply(convert_type)
    df["Percentage of rows impacted"] = df["Percentage_of_valid_rows"].apply(lambda x: 100 - x)
    df["Number of impacted rows"] = df["Rows"].apply(lambda x: len(x))
    df["Wrong rows's index"] = df["Rows"].astype(str).str.replace(r'[\[\]]', '', regex=True)

    df["Description"] = df.apply(
        lambda x: (
            "Selecting one answer prevents the selection of any other options."
            if x["Logic Type"] == "Exclusive"
            else "Minimum and/or maximum limits on the number of choices a respondent can select."
            if x["Logic Type"] == "Selection Limit Control"
            else x["Detail"] if pd.notna(x["Detail"]) else ""
        ),
        axis=1
    )

    df["Columns"] = df["Dataframe"].apply(lambda x: ", ".join(map(str, pd.DataFrame(x).columns)))

    df["Supported"] = df["is_supported"] .apply(lambda x: "No" if x == False else "Yes")

    return df[["Logic Type", "Description", "Columns", "Percentage of rows impacted", "Number of impacted rows", "Wrong rows's index", "Supported"]]


def export_to_excel(df, filename="outputs/template.xlsx"):
    workbook = xlsxwriter.Workbook(filename, {'nan_inf_to_errors': True})
    worksheet = workbook.add_worksheet('Table')

    # === Formats ===
    banner_format = workbook.add_format({
        'bold': True,
        'font_name': 'Roboto',
        'font_size': 14,
        'bg_color': '#143126',
        'font_color': 'white'
    })

    column_header_format = workbook.add_format({
        'bold': True,
        'font_name': 'Roboto',
        'font_size': 11,
        'bg_color': '#ccffcb',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })

    data_format = workbook.add_format({
        'font_name': 'Roboto',
        'font_size': 10,
        'align': 'left',
        'valign': 'vcenter',
        'border': 1,
        'text_wrap': True,
        'bold': False  # Ensure data is NOT bold
    })

    title_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'font_name': 'Roboto',
        'align': 'left'
    })

    summary_format = workbook.add_format({
        'bold': True,
        'font_size': 11,
        'font_name': 'Roboto',
        'align': 'right'
    })

    # === Banner/Header Section ===
    worksheet.set_row(0, 20, banner_format)
    worksheet.set_row(1, 20, banner_format)
    worksheet.set_row(2, 20, banner_format)
    worksheet.merge_range("B1:E2", "Logic Attainement", banner_format)
    worksheet.merge_range("B3:E3", "REPORT", banner_format)

    start_row = 7  # Data starts on Excel row 11
    start_col = 1   # Start at column B

    # === Title above table ===
    worksheet.write(start_row - 2, start_col, "List of limitations:", title_format)

    # === Write headers ===
    for col_num, col_name in enumerate(df.columns):
        worksheet.write(start_row, start_col + col_num, col_name, column_header_format)

    # === Write data rows with dynamic row height ===
    for row_num, row in df.iterrows():
        max_lines = 1
        for col_num, (col_name, cell_value) in enumerate(row.items()):
            cell_str = str(cell_value) if pd.notna(cell_value) else ""
            worksheet.write(start_row + row_num + 1, start_col + col_num, cell_str, data_format)

            if col_name != "Wrong rows's index":
                est_lines = math.ceil(len(cell_str) / 30)
                max_lines = max(max_lines, est_lines)

        row_height = max(20, max_lines * 15)
        worksheet.set_row(start_row + row_num + 1, row_height)

    # === Set column widths ===
    worksheet.set_column(start_col, start_col + len(df.columns) - 1, 30)

    # === Calculate unique index count from 'Wrong rows's index' ===
    index_series = df["Wrong rows's index"]
    all_indexes = set()

    for val in index_series:
        numbers = re.findall(r'\d+', str(val))
        all_indexes.update(map(int, numbers))

    total_index_count = len(all_indexes)

    # === Write summary at bottom-right of last VISIBLE column ===
    summary_row = start_row + len(df) + 2
    visible_cols = [col for col in df.columns if col != "Wrong rows's index"]
    last_visible_col_index = len(visible_cols) - 1
    summary_col = start_col - 1 + last_visible_col_index
    worksheet.write(summary_row, summary_col, f"Number of rows affected: {total_index_count}", summary_format)

    # === Hide "Wrong rows's index" column ===
    wrong_index_col = df.columns.get_loc("Wrong rows's index")
    worksheet.set_column(start_col + wrong_index_col, start_col + wrong_index_col, None, None, {'hidden': True})

    workbook.close()

