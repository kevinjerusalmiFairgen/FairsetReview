import pandas as pd
from notion_client import Client
from collections import OrderedDict
import re


TOKEN = "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"


def extract_property(prop):
    """
    Extract a display value from a Notion property based on its type.
    """
    prop_type = prop.get("type")
    if prop_type == "title":
        return " ".join(item.get("plain_text", "") for item in prop.get("title", []))
    elif prop_type == "rich_text":
        return " ".join(item.get("plain_text", "") for item in prop.get("rich_text", []))
    elif prop_type == "number":
        return prop.get("number")
    elif prop_type == "select":
        select_obj = prop.get("select")
        return select_obj.get("name") if select_obj else None
    elif prop_type == "multi_select":
        return ", ".join(item.get("name", "") for item in prop.get("multi_select", []))
    elif prop_type == "date":
        date_obj = prop.get("date")
        return date_obj.get("start") if date_obj else None
    elif prop_type == "checkbox":
        return prop.get("checkbox")
    elif prop_type == "url":
        return prop.get("url")
    elif prop_type == "formula":
        # Formula can return different types; here we try to extract a string representation.
        formula_val = prop.get("formula", {})
        for key in ["string", "number", "boolean"]:
            if key in formula_val and formula_val[key] is not None:
                return formula_val[key]
        return None
    elif prop_type == "relation":
        # Relation returns a list of objects (each with an "id")
        return ", ".join(item.get("id", "") for item in prop.get("relation", []))
    else:
        # Fallback: return the raw property
        return prop.get(prop_type)


def extract_properties(properties):
    """
    Given the properties dict from a page, extract a flat dict of property names and their display values.
    """
    extracted = OrderedDict()
    for key, prop in properties.items():
        extracted[key] = extract_property(prop)
    return extracted


def cleaning_lists(target_str):
    if not isinstance(target_str, str):
        return target_str
    if target_str.startswith('['):
        clean_str = target_str.strip("[]")
        list_items = re.findall(r"/?\s*['\"]([^'\"]+)['\"]\s*/?", clean_str)
        return list_items
    else:
        return target_str


def extractAndPreprocessing(database_id, index_order=None):
    token = TOKEN
    notion = Client(auth=token)
    response = notion.databases.query(database_id=database_id)
    extracted_pages = [extract_properties(page["properties"]) for page in response["results"]]
    df = pd.DataFrame(extracted_pages)[["Target", "Source", "Constraint", "B/F Relationship", "Comment", "Is Implemented", "Custom Query", "ID"]]
    df["Target"] = df["Target"].apply(cleaning_lists)
    df["Source"] = df["Source"].apply(cleaning_lists)
    df['ID'] = df['ID'].apply(lambda x: x['number'] if isinstance(x, dict) else x)

    if index_order is not None:
        df = df.set_index('ID')
        df = df.loc[index_order]
    return df


def createJSONs(df):
    constraints_json = {
        "BF_SS": [],
        "BF_SM": [],
        "BF_MS": [],
        "BF_MM": [],
        "uniqueness": [],
        "count": [],
        "recodings": [],
        "NOTAs": [],
        "AOTAS": [],
        "parallel piping": [],
        "custom": []
    }

    structure_json = {
        "recodings": [],
        "multiSelect": [],
        "typeOfNan": [],
    }

    for index, row in df.iterrows():
        def normalize_quotes(text):
            # Replace curly single and double quotes with straight quotes
            if isinstance(text, str):
                return text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
            elif isinstance(text, list):
                return [normalize_quotes(item) for item in text]
            return text  # Return as-is if not string or list

        # Applying normalization
        row["Source"] = normalize_quotes(row["Source"])
        row["Target"] = normalize_quotes(row["Target"])

        
        if row["Custom Query"] != '':
            constraints_json["custom"].append([
                row["Constraint"],
                row["Comment"],
                row["Custom Query"]
            ])
        else:
            if row["Constraint"] in ["Block", "Force", "Block/Force"]:
                if not row["B/F Relationship"]:
                    print(f"Row #{row["ID"]} not valid: missing relationship for Block/Force")
                elif row["B/F Relationship"] == "Single to Single":
                    constraints_json["BF_SS"].append([
                        row["Source"],
                        row["Target"],
                        row["Comment"],
                        row["Constraint"].lower().replace("/", "_"),
                        row["Is Implemented"]
                    ])
                elif row["B/F Relationship"] == "Single to Multi":
                    constraints_json["BF_SM"].append([
                        row["Source"],
                        row["Target"],
                        row["Comment"],
                        row["Constraint"].lower().replace("/", "_"),
                        row["Is Implemented"]
                    ])
                elif row["B/F Relationship"] == "Multi to Single":
                    constraints_json["BF_MS"].append([
                        row["Source"],
                        row["Target"],
                        row["Comment"],
                        row["Constraint"].lower().replace("/", "_"),
                        row["Is Implemented"]
                    ])
                else:
                    print(f"Row #{row["ID"]} not valid: wrong relationship for Block/Force")
                    
            elif row["Constraint"] == "Parallel Piping":
                constraints_json["BF_MM"].append([
                    row["Source"],
                    row["Target"],
                    [],
                    row["Comment"],
                    row["Constraint"].lower().replace("/", "_"),
                    row["Is Implemented"]
                ])
            elif row["Constraint"] == "Uniqueness":
                constraints_json["uniqueness"].append([
                    row["Target"],
                    row["Is Implemented"]
                ])
            elif row["Constraint"] == "Count":
                constraints_json["count"].append([
                    row["Target"],
                    "",
                    row["Is Implemented"]
                ])
            elif row["Constraint"] == "Recoding":
                if not row["B/F Relationship"]:
                    print(f"Row #{row["ID"]} not valid: missing relationship for Recoding")
                elif row["B/F Relationship"] == "Single to Single":
                    mode = "SS"
                    structure_json["recodings"].append(
                        {
                            "id": index,
                            "name": index,
                            "recode": row["Target"],
                            "codes": row["Source"]
                        }
                    )
                elif row["B/F Relationship"] == "Single to Multi":
                    mode = "SM"
                elif row["B/F Relationship"] == "Multi to Single":
                    mode = "MS"
                    structure_json["recodings"].append(
                        {
                            "id": index,
                            "name": index,
                            "recode": row["Target"],
                            "codes": row["Source"]
                        }
                    )
                elif row["B/F Relationship"] == "Multi to Multi":
                    mode = "MM"
                else:
                    print(f"Row #{row["ID"]} not valid: wrong relationship for Recoding")
                constraints_json["recodings"].append([
                    row["Source"],
                    row["Target"],
                    mode,
                    row["Comment"],
                    row["Is Implemented"]
                ])
                structure_json["recodings"].append([
                        {
                            "id": index,
                            "name": index,
                            "recode": row["Target"],
                            "codes": row["Source"]
                        }
                ])
            elif row["Constraint"] == "None of the above":
                constraints_json["NOTAs"].append([
                    row["Source"],
                    row["Target"],
                    [],
                    row["Is Implemented"]
                ])
            elif row["Constraint"] == "All of the above":
                constraints_json["AOTAs"].append([
                    row["Source"],
                    row["Target"],
                    [],
                    row["Is Implemented"]
                ])
            elif row["Constraint"] == "MultiSelect":
                structure_json["multiSelect"].append(
                    {
                        "id": index,
                        "name": index,
                        "columns": row["Target"],
                        "multiSelect_id": index
                    }
                )

    return constraints_json, structure_json