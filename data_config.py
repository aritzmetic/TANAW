import os
import pandas as pd
import re

def get_dataset_path(filename="Cleaned_School_DataSet.csv"):
    return os.path.join(os.path.dirname(__file__), 'static', filename)

def fetch_enrollment_records_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    except FileNotFoundError as e:
        print(f"Error: File not found at {file_path}: {e}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def fetch_summary_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)

        male_cols = [col for col in df.columns if re.search(r'\bmale\b', col, re.IGNORECASE)]
        female_cols = [col for col in df.columns if re.search(r'\bfemale\b', col, re.IGNORECASE)]

        for col in male_cols + female_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        total_male = df[male_cols].sum().sum()
        total_female = df[female_cols].sum().sum()
        total_enrollments = total_male + total_female

        is_school_level = 'BEIS School ID' in df.columns
        number_of_schools = df['BEIS School ID'].nunique() if is_school_level else None
        number_of_year_levels = 13

        # Try to find the 'Region' column regardless of casing
        region_col = next((col for col in df.columns if col.strip().lower() == 'region'), None)
        number_of_regions = df[region_col].nunique() if region_col else 0

        summary = {
            'totalEnrollments': int(total_enrollments),
            'maleEnrollments': int(total_male),
            'femaleEnrollments': int(total_female),
            'numberOfYearLevels': number_of_year_levels,
            'regionsWithSchools': int(number_of_regions)
        }

        if is_school_level:
            summary['numberOfSchools'] = int(number_of_schools)

        return summary

    except Exception as e:
        print(f"Error processing summary data: {e}")
        return {}