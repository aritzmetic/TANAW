import os
import pandas as pd
from datetime import datetime

def clean_data(file_path):

    df = pd.read_csv(file_path)

    cleaned_files_directory = os.path.join(os.path.dirname(__file__), 'cleaned_files')
    os.makedirs(cleaned_files_directory, exist_ok=True)

    df_cleaned = df.iloc[3:].reset_index(drop=True)
    df_cleaned.columns = df_cleaned.iloc[0]
    df_cleaned = df_cleaned[1:].reset_index(drop=True)
    df_cleaned = df_cleaned.apply(pd.to_numeric, errors='ignore')

    special_cases = {
        r'\bES\b': 'ELEMENTARY SCHOOL',
        'E/S': 'ELEMENTARY SCHOOL',
        r'\bELEM.\b': 'ELEMENTARY SCHOOL',
        r'\bNHS\b': 'NATIONAL HIGH SCHOOL',
        r'\bHS\b': 'HIGH SCHOOL',
        r'\bCES\b': 'CENTRAL ELEMENTARY SCHOOL',
        r'\bSCH.\b': 'SCHOOL',
        'Incorporated': 'INC.',
        r'\bMEM.\b': 'MEMORIAL',
        r'\bCS\b': 'CENTRAL SCHOOL',
        r'\bPS\b': 'PRIMARY SCHOOL',
        'P/S': 'PRIMARY SCHOOL',
        r'\bLC\b': 'LEARNING CENTER',
        'BARANGAY': 'BRGY. ',
        'POBLACION': 'POB. ',
        'STREET': 'ST. ',
        'BUILDING': 'BLDG. ',
        'BLOCK': 'BLK. ',
        'PUROK': 'PRK. ',
        'AVENUE': 'AVE. ',
        'ROAD': 'RD. ',
        'PACKAGE': 'PKG. ',
        'PHASE': 'PH. ',
        r'\s*,\s*': ', ',
        r'\s{2,}': ' '
    }

    columns_to_format = ['School Name', 'Street Address', 'Province', 'Municipality', 'Barangay']
    df_cleaned[columns_to_format] = df_cleaned[columns_to_format].apply(
        lambda x: x.str.replace('#', '', regex=False)
                    .str.replace(r'^[-:]', '', regex=True)
                    .str.strip()
                    .str.upper()
                    .replace(special_cases, regex=True)
    )

    columns_to_format = ['Street Address', 'Barangay']
    df_cleaned[columns_to_format] = (
        df_cleaned[columns_to_format]
        .replace(['N/A', 'N.A.', 'N / A', 'NA', 'NONE', 'NULL', 'NOT APPLICABLE', '', '0', '_', '=', '.', '-----'], pd.NA)
        .replace(r'^[\s\W_]+$', pd.NA, regex=True)
        .fillna("UNKNOWN")
    )

    non_enrollment_cols = [
        'Region', 'Division', 'District', 'BEIS School ID', 'School Name',
        'Street Address', 'Province', 'Municipality', 'Legislative District',
        'Barangay', 'Sector', 'School Subclassification', 'School Type', 'Modified COC'
    ]

    enrollment_cols = [col for col in df_cleaned.columns if col not in non_enrollment_cols]
    max_threshold = 5000

    unrealistic_data = df_cleaned[(
        df_cleaned[enrollment_cols] < 0).any(axis=1) |
        (df_cleaned[enrollment_cols] % 1 != 0).any(axis=1) |
        (df_cleaned[enrollment_cols] > max_threshold).any(axis=1)
    ]

    df_cleaned = df_cleaned.drop(unrealistic_data.index)

    cleaned_filename = f"cleaned_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    cleaned_path = os.path.join(cleaned_files_directory, cleaned_filename)

    if isinstance(df_cleaned, pd.DataFrame):
        df_cleaned.to_csv(cleaned_path, index=False)
        return cleaned_path 
    else:
        raise ValueError("DataFrame was not cleaned properly.")
    