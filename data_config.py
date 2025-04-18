import os
import pandas as pd

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

        total_male = df.filter(like='Male').sum().sum()
        total_female = df.filter(like='Female').sum().sum()
        total_enrollments = total_male + total_female
        number_of_schools = df['BEIS School ID'].nunique()
        regions_with_schools = df['Region'].nunique()
        number_of_year_levels = 13  # K to G12
        

        return {
            'totalEnrollments': int(total_enrollments),
            'maleEnrollments': int(total_male),
            'femaleEnrollments': int(total_female),
            'numberOfSchools': int(number_of_schools),
            'regionsWithSchools': int(regions_with_schools),
            'numberOfYearLevels': int(number_of_year_levels)
        }
    except Exception as e:
        print(f"Error processing summary data: {e}")
        return {}
