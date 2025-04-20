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

        # Identify male and female columns
        male_cols = [col for col in df.columns if re.search(r'\bmale\b', col, re.IGNORECASE)]
        female_cols = [col for col in df.columns if re.search(r'\bfemale\b', col, re.IGNORECASE)]

        for col in male_cols + female_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        df['TotalMale'] = df[male_cols].sum(axis=1)
        df['TotalFemale'] = df[female_cols].sum(axis=1)
        df['TotalEnrollment'] = df['TotalMale'] + df['TotalFemale']

        total_male = df['TotalMale'].sum()
        total_female = df['TotalFemale'].sum()
        total_enrollments = total_male + total_female

        # Try to find the 'Region' column
        region_col = next((col for col in df.columns if col.strip().lower() == 'region'), None)
        number_of_regions = df[region_col].nunique() if region_col else 0

        # Determine if school-level
        is_school_level = 'BEIS School ID' in df.columns and 'School Name' in df.columns
        number_of_schools = df['BEIS School ID'].nunique() if is_school_level else None
        number_of_year_levels = 13

        # Top schools logic
        if is_school_level:
            df['UniqueSchool'] = df['School Name'] + " (" + df['BEIS School ID'].astype(str) + ")"
            top_schools = (
                df.groupby('UniqueSchool')['TotalEnrollment']
                .sum().sort_values(ascending=False).head(5)
                .astype(int).to_dict()
            )
        else:
            top_schools = {}

        # Enrollment by Region (count of schools/rows)
        enrollment_by_region = df[region_col].value_counts().to_dict() if region_col else {}

        # Enrollment by year level (sums per grade)
        grade_columns = {
            'Kindergarten': ['K Male', 'K Female'],
            'Grade 1': ['G1 Male', 'G1 Female'],
            'Grade 2': ['G2 Male', 'G2 Female'],
            'Grade 3': ['G3 Male', 'G3 Female'],
            'Grade 4': ['G4 Male', 'G4 Female'],
            'Grade 5': ['G5 Male', 'G5 Female'],
            'Grade 6': ['G6 Male', 'G6 Female'],
            'Grade 7': ['G7 Male', 'G7 Female'],
            'Grade 8': ['G8 Male', 'G8 Female'],
            'Grade 9': ['G9 Male', 'G9 Female'],
            'Grade 10': ['G10 Male', 'G10 Female'],
            'Grade 11': [
                'G11 ACAD - ABM Male', 'G11 ACAD - ABM Female',
                'G11 ACAD - HUMSS Male', 'G11 ACAD - HUMSS Female',
                'G11 ACAD STEM Male', 'G11 ACAD STEM Female',
                'G11 ACAD GAS Male', 'G11 ACAD GAS Female',
                'G11 ACAD PBM Male', 'G11 ACAD PBM Female',
                'G11 TVL Male', 'G11 TVL Female',
                'G11 SPORTS Male', 'G11 SPORTS Female',
                'G11 ARTS Male', 'G11 ARTS Female'
            ],
            'Grade 12': [
                'G12 ACAD - ABM Male', 'G12 ACAD - ABM Female',
                'G12 ACAD - HUMSS Male', 'G12 ACAD - HUMSS Female',
                'G12 ACAD STEM Male', 'G12 ACAD STEM Female',
                'G12 ACAD GAS Male', 'G12 ACAD GAS Female',
                'G12 ACAD PBM Male', 'G12 ACAD PBM Female',
                'G12 TVL Male', 'G12 TVL Female',
                'G12 SPORTS Male', 'G12 SPORTS Female',
                'G12 ARTS Male', 'G12 ARTS Female'
            ]
        }

        enrollment_by_year_level = {}
        for grade, columns in grade_columns.items():
            valid_cols = [col for col in columns if col in df.columns]
            enrollment_by_year_level[grade] = int(df[valid_cols].sum().sum()) if valid_cols else 0

        # Average Enrollment Per Region
        avg_enrollment_per_region = (
            df.groupby(region_col)['TotalEnrollment'].mean()
            .round(2).astype(int).to_dict()
            if region_col else {}
        )

        # Gender Ratio Per Region
        if region_col:
            grouped = df.groupby(region_col)[['TotalMale', 'TotalFemale']].sum()
            grouped['TotalEnrollment'] = grouped['TotalMale'] + grouped['TotalFemale']
            grouped['MalePercentage'] = (grouped['TotalMale'] / grouped['TotalEnrollment']) * 100
            gender_ratio_by_region = grouped['MalePercentage'].round(2).to_dict()
        else:
            gender_ratio_by_region = {}

        # SHS Enrollment by Strand
        shs_strands = {
            'ABM': ['G11 ACAD - ABM Male', 'G11 ACAD - ABM Female', 'G12 ACAD - ABM Male', 'G12 ACAD - ABM Female'],
            'HUMSS': ['G11 ACAD - HUMSS Male', 'G11 ACAD - HUMSS Female', 'G12 ACAD - HUMSS Male', 'G12 ACAD - HUMSS Female'],
            'STEM': ['G11 ACAD STEM Male', 'G11 ACAD STEM Female', 'G12 ACAD STEM Male', 'G12 ACAD STEM Female'],
            'GAS': ['G11 ACAD GAS Male', 'G11 ACAD GAS Female', 'G12 ACAD GAS Male', 'G12 ACAD GAS Female'],
            'PBM': ['G11 ACAD PBM Male', 'G11 ACAD PBM Female', 'G12 ACAD PBM Male', 'G12 ACAD PBM Female'],
            'TVL': ['G11 TVL Male', 'G11 TVL Female', 'G12 TVL Male', 'G12 TVL Female'],
            'SPORTS': ['G11 SPORTS Male', 'G11 SPORTS Female', 'G12 SPORTS Male', 'G12 SPORTS Female'],
            'ARTS': ['G11 ARTS Male', 'G11 ARTS Female', 'G12 ARTS Male', 'G12 ARTS Female']
        }

        shs_enrollment_by_strand = {}
        for strand, cols in shs_strands.items():
            valid_cols = [col for col in cols if col in df.columns]
            shs_enrollment_by_strand[strand] = int(df[valid_cols].sum().sum()) if valid_cols else 0

        # Final summary
        summary = {
            'totalEnrollments': int(total_enrollments),
            'maleEnrollments': int(total_male),
            'femaleEnrollments': int(total_female),
            'regionsWithSchools': int(number_of_regions),
            'numberOfYearLevels': number_of_year_levels,
            'topSchools': top_schools,
            "enrollmentByRegion": enrollment_by_region,
            'enrollmentByYearLevel': enrollment_by_year_level,
            'averageEnrollmentPerRegion': avg_enrollment_per_region,
            'genderRatioByRegion': gender_ratio_by_region,
            'shsEnrollmentByStrand': shs_enrollment_by_strand
        }

        if is_school_level:
            summary['numberOfSchools'] = int(number_of_schools)

        return summary

    except Exception as e:
        print(f"Error processing summary data: {e}")
        return {}
