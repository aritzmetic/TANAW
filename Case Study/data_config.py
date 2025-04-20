import os
import pandas as pd
import re

# Returns the full path of the dataset CSV file located in /static/
def get_dataset_path(filename="Cleaned_School_DataSet.csv"):
    return os.path.join(os.path.dirname(__file__), 'static', filename)

# Reads CSV file and returns a list of enrollment records as dictionaries
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

# Fetches and summarizes various metrics from the enrollment CSV data
def fetch_summary_data_from_csv(file_path):
    try:
        df = pd.read_csv(file_path)
        df['UniqueSchool'] = df['School Name'] + " (" + df['BEIS School ID'].astype(str) + ")"
        df['TotalEnrollment'] = df.filter(like='Male').sum(axis=1) + df.filter(like='Female').sum(axis=1)

        # Compute total counts
        total_male = df.filter(like='Male').sum().sum()
        total_female = df.filter(like='Female').sum().sum()
        total_enrollments = total_male + total_female
        number_of_schools = df['BEIS School ID'].nunique()
        regions_with_schools = df['Region'].nunique()
        number_of_year_levels = 13 
        top_schools = (df.groupby('UniqueSchool')['TotalEnrollment'].sum().sort_values(ascending=False).head(5).astype(int).to_dict())
        total_divisions = df['Division'].nunique()
        total_municipalities = df['Municipality'].nunique()

        # Enrollment count per region
        if "Region" in df.columns:
            enrollment_by_region = df["Region"].value_counts().to_dict()
        else:
            enrollment_by_region = {}
        
        # Define columns for each year level
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

        # Enrollment per grade level
        enrollment_by_year_level = {}
        for grade, columns in grade_columns.items():
            valid_cols = [col for col in columns if col in df.columns]
            enrollment_by_year_level[grade] = int(df[valid_cols].sum().sum()) if valid_cols else 0

        # Average enrollment per region
        if "Region" in df.columns and "TotalEnrollment" in df.columns:
            avg_enrollment_per_region = (
                df.groupby('Region')['TotalEnrollment']
                .mean()
                .round(2)
                .astype(int)
                .to_dict()
            )
        else:
            avg_enrollment_per_region = {}

        # Calculate total male and female per row for gender breakdown
        df['TotalMale'] = df.filter(like='Male').sum(axis=1)
        df['TotalFemale'] = df.filter(like='Female').sum(axis=1)

        # Group by region and calculate gender ratios
        grouped = df.groupby('Region')[['TotalMale', 'TotalFemale']].sum()
        grouped['TotalEnrollment'] = grouped['TotalMale'] + grouped['TotalFemale']
        grouped['MalePercentage'] = (grouped['TotalMale'] / grouped['TotalEnrollment']) * 100
        gender_ratio_by_region = grouped['MalePercentage'].round(2).to_dict()

        # SHS strand enrollment columns
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

        # SHS enrollment per strand
        shs_enrollment_by_strand = {}
        for strand, cols in shs_strands.items():
            valid_cols = [col for col in cols if col in df.columns]
            shs_enrollment_by_strand[strand] = int(df[valid_cols].sum().sum()) if valid_cols else 0

        # Enrollment by School Sector
        if 'Sector' in df.columns:
            enrollment_by_sector_df = df.groupby('Sector')['TotalEnrollment'].sum()
            enrollment_by_sector = {
                sector: int(enrollment)
                for sector, enrollment in enrollment_by_sector_df.items()
            }
        else:
            enrollment_by_sector = {}

        # Return all summary statistics
        return {
            'totalEnrollments': int(total_enrollments),
            'maleEnrollments': int(total_male),
            'femaleEnrollments': int(total_female),
            'numberOfDivisions': int(total_divisions),
            'numberOfMunicipalities': int(total_municipalities),
            'numberOfSchools': int(number_of_schools),
            'regionsWithSchools': int(regions_with_schools),
            'numberOfYearLevels': int(number_of_year_levels),
            'topSchools': top_schools,
            "enrollmentByRegion": enrollment_by_region,
            'enrollmentByYearLevel': enrollment_by_year_level,
            'averageEnrollmentPerRegion': avg_enrollment_per_region,
            'genderRatioByRegion': gender_ratio_by_region,
            'shsEnrollmentByStrand': shs_enrollment_by_strand,
            'enrollmentBySector': enrollment_by_sector 
        }
    except Exception as e:
        print(f"Error processing summary data: {e}")
        return {}
