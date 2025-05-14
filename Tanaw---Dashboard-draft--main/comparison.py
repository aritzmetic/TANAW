# comparison.py
import pandas as pd
import re
import os
import numpy as np 

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_MANAGEMENT_FOLDER = os.path.join(CURRENT_FILE_DIR, 'data_management')

def normalize_region_name(region_name):
    if pd.isna(region_name) or not isinstance(region_name, str):
        return "Unknown"
    name = region_name.strip().upper()
    if "MIMAROPA" in name: return "MIMAROPA"
    if name in ["REGION IV-A", "CALABARZON"]: return "REGION 04A (CALABARZON)"
    if name == "REGION IV-B": return "MIMAROPA"
    if name in ["REGION XII", "SOCCSKSARGEN"]: return "REGION 12 (SOCCSKSARGEN)"
    if "NATIONAL CAPITAL" in name or name == "NCR": return "NATIONAL CAPITAL REGION"
    if name == "AUTONOMOUS REGION IN MUSLIM MINDANAO (ARMM)": return "ARMM"
    if name == "BANGSAMORO AUTONOMOUS REGION IN MUSLIM MINDANAO (BARMM)": return "BARMM"
    if name == "CORDILLERA ADMINISTRATIVE REGION" or name == "CAR": return "CAR"
    
    match_roman = re.match(r"REGION\s*([IVXLCDM]+)$", name)
    if match_roman:
        roman_numeral = match_roman.group(1)
        roman_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        int_val = 0; prev_val = 0
        for char_val in reversed(roman_numeral):
            val = roman_map.get(char_val, 0)
            if val < prev_val: int_val -= val
            else: int_val += val
            prev_val = val
        return f"REGION {int_val:02d}"
    match_arabic = re.match(r"REGION\s*(\d+)$", name)
    if match_arabic: return f"REGION {int(match_arabic.group(1)):02d}"
    return name

def find_available_datasets():
    datasets = {}
    years = []
    if not os.path.exists(DATA_MANAGEMENT_FOLDER):
        print(f"Comparison Module: Data management folder not found at {DATA_MANAGEMENT_FOLDER}")
        return {}, []
    for year_dir in os.listdir(DATA_MANAGEMENT_FOLDER):
        year_path = os.path.join(DATA_MANAGEMENT_FOLDER, year_dir)
        if os.path.isdir(year_path) and "-" in year_dir:
            for filename in os.listdir(year_path):
                if filename.lower().endswith(".csv"):
                    datasets[year_dir] = os.path.join(year_path, filename)
                    years.append(year_dir)
                    break
    years.sort(reverse=True)
    return {year: datasets[year] for year in years if year in datasets}, years

def get_school_size_category(enrollment):
    if pd.isna(enrollment) or enrollment is None: return "Unknown"
    if enrollment <= 50: return "Very Small (<=50)"
    if enrollment <= 200: return "Small (51-200)"
    if enrollment <= 500: return "Medium (201-500)"
    if enrollment <= 1000: return "Large (501-1000)"
    return "Very Large (>1000)"

def extract_relevant_data(file_path):
    if not file_path or not os.path.exists(file_path):
        print(f"Comparison Module: File not found at {file_path}")
        return None
    try:
        df = pd.read_csv(file_path, low_memory=False)
        enrollment_patterns = ['K ', 'G1 ', 'G2 ', 'G3 ', 'G4 ', 'G5 ', 'G6 ', 'G7 ', 'G8 ', 'G9 ', 'G10 ', 'G11 ', 'G12 ', 'Elem NG ', 'JHS NG ']
        enrollment_cols = [c for c in df.columns if any(c.startswith(p) for p in enrollment_patterns) and (' Male' in c or ' Female' in c)]

        if not enrollment_cols:
            print(f"Warning: No enrollment columns found in {file_path}.")
            return None 

        for col in enrollment_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df[enrollment_cols] = df[enrollment_cols].fillna(0).astype(int)

        id_cat_cols = ['BEIS School ID', 'Region', 'Division', 'Sector', 'School Name']
        for col in id_cat_cols:
            df[col] = df[col].astype(str).fillna('Unknown') if col in df.columns else 'Unknown'
        
        if 'Region' in df.columns:
            df['Region'] = df['Region'].apply(normalize_region_name)

        df['Total Enrollment per School'] = df[enrollment_cols].sum(axis=1)
        total_enrollment_overall = int(df['Total Enrollment per School'].sum())

        enrollment_by_sector = {str(k): int(v) for k, v in df.groupby('Sector')['Total Enrollment per School'].sum().to_dict().items()} if 'Sector' in df.columns else {}

        elem_cols = [c for c in enrollment_cols if any(p in c for p in ['K ','G1 ','G2 ','G3 ','G4 ','G5 ','G6 ','Elem NG ']) and c in df.columns]
        jhs_cols = [c for c in enrollment_cols if any(p in c for p in ['G7 ','G8 ','G9 ','G10 ','JHS NG ']) and c in df.columns]
        shs_cols = [c for c in enrollment_cols if any(p in c for p in ['G11 ','G12 ']) and c in df.columns]

        total_shs_enrollment = int(df[shs_cols].sum().sum()) if shs_cols else 0 # For new KPI

        enrollment_by_level = {
            'Elementary': int(df[elem_cols].sum().sum()) if elem_cols else 0,
            'Junior High': int(df[jhs_cols].sum().sum()) if jhs_cols else 0,
            'Senior High': total_shs_enrollment # Use already calculated total_shs_enrollment
        }
        
        gpi_by_level = {}
        for level_name, level_cols_list in {'Elementary': elem_cols, 'Junior High': jhs_cols, 'Senior High': shs_cols}.items():
            if not level_cols_list: gpi_by_level[level_name] = None; continue
            male_sum = int(df[[c for c in level_cols_list if ' Male' in c]].sum().sum())
            female_sum = int(df[[c for c in level_cols_list if ' Female' in c]].sum().sum())
            gpi_by_level[level_name] = float(female_sum / male_sum) if male_sum > 0 else None

        total_male = int(df[[c for c in enrollment_cols if ' Male' in c]].sum().sum())
        total_female = int(df[[c for c in enrollment_cols if ' Female' in c]].sum().sum())
        gender_parity_index_overall = float(total_female / total_male) if total_male > 0 else None # Use None for N/A

        num_schools = 0
        school_size_distribution = {}
        if 'BEIS School ID' in df.columns:
            valid_schools_df = df[df['BEIS School ID'].notna() & (df['BEIS School ID'] != 'Unknown')]
            if not valid_schools_df.empty:
                num_schools = int(valid_schools_df['BEIS School ID'].nunique())
                if num_schools > 0:
                    school_enrollments = valid_schools_df.groupby('BEIS School ID')['Total Enrollment per School'].first()
                    school_enrollments = school_enrollments[school_enrollments > 0] # Consider only schools with enrollment
                    if not school_enrollments.empty:
                        df_school_sizes = pd.DataFrame({'enrollment': school_enrollments})
                        df_school_sizes['category'] = df_school_sizes['enrollment'].apply(get_school_size_category)
                        school_size_distribution_raw = df_school_sizes['category'].value_counts().to_dict()
                        school_size_distribution = {str(k): int(v) for k,v in school_size_distribution_raw.items()}
                    else: num_schools = 0 # If no schools have enrollment after filtering
            else: num_schools = 0
        
        # Avg school size is no longer a primary KPI, but can be derived if needed:
        # avg_school_size = float(total_enrollment_overall / num_schools) if num_schools > 0 else None

        enrollment_by_region = {str(k): int(v) for k,v in df.groupby('Region')['Total Enrollment per School'].sum().sort_values(ascending=False).to_dict().items()} if 'Region' in df.columns else {}
        
        shs_strand_enrollment = {}
        schools_offering_shs_strands = {}
        if shs_cols:
            shs_pattern = re.compile(r'(G11|G12)\s+(.+?)\s+(Male|Female)$')
            processed_schools_for_strand = {}
            temp_shs_strand_enrollment = {}
            for col in shs_cols:
                match = shs_pattern.match(col)
                if match:
                    strand_name = str(match.group(2).replace(" Strand", "").replace(" Track", "").strip().upper())
                    if not strand_name: continue
                    temp_shs_strand_enrollment[strand_name] = temp_shs_strand_enrollment.get(strand_name, 0) + int(df[col].sum())
                    if strand_name not in processed_schools_for_strand: processed_schools_for_strand[strand_name] = set()
                    processed_schools_for_strand[strand_name].update(df[(df[col] > 0) & (df['BEIS School ID'] != 'Unknown')]['BEIS School ID'].unique()) # Ensure valid school IDs
            
            shs_strand_enrollment = dict(sorted(temp_shs_strand_enrollment.items(), key=lambda item: item[1], reverse=True))
            for k, v_set in processed_schools_for_strand.items():
                schools_offering_shs_strands[k] = len(v_set) 
            schools_offering_shs_strands = dict(sorted(schools_offering_shs_strands.items(), key=lambda item: item[1], reverse=True))

        return {
            'total_enrollment': total_enrollment_overall, 'enrollment_by_sector': enrollment_by_sector,
            'enrollment_by_level': enrollment_by_level, 'gpi_by_level': gpi_by_level,
            'total_male': total_male, 'total_female': total_female,
            'gender_parity_index_overall': gender_parity_index_overall,
            'num_schools': num_schools, 
            'school_size_distribution': school_size_distribution, 
            'total_shs_enrollment': total_shs_enrollment, # Added for new KPI
            'enrollment_by_region': enrollment_by_region,
            'shs_strand_enrollment': shs_strand_enrollment,
            'schools_offering_shs_strands': schools_offering_shs_strands
        }
    except Exception as e:
        print(f"Comparison Module: Error processing file {file_path}: {e}")
        import traceback; traceback.print_exc()
        return None

def prepare_comparison_charts_data(file_path1, file_path2, year1_label, year2_label):
    data1 = extract_relevant_data(file_path1)
    data2 = extract_relevant_data(file_path2)

    if not data1 or not data2:
        error_msg = "Could not process data for comparison. "
        if not data1 and file_path1: error_msg += f"Issue with {year1_label} data. "
        if not data2 and file_path2: error_msg += f"Issue with {year2_label} data. "
        return {'error': error_msg.strip(), 'year1_label': str(year1_label), 'year2_label': str(year2_label)}

    results = {
        'year1_label': str(year1_label), 'year2_label': str(year2_label),
        'summary_kpis': [], 'enrollment_by_sector': None, 'enrollment_by_level': None,
        'gpi_by_level_comparison': None, 'gender_distribution': None,
        'regional_comparison': None,
        'shs_strand_enrollment_comparison': None,
        'schools_offering_shs_strands_change': None, 
        'school_size_distribution_comparison': None
    }

    # KPIs
    shs_enroll1 = data1.get('total_shs_enrollment', 0)
    shs_enroll2 = data2.get('total_shs_enrollment', 0)
    shs_change_raw, shs_change_fmt = "N/A", "N/A"
    if isinstance(shs_enroll1, (int, float)) and isinstance(shs_enroll2, (int, float)):
        if shs_enroll1 != 0:
            shs_change_raw = float(((shs_enroll2 - shs_enroll1) / shs_enroll1 * 100))
            shs_change_fmt = f"{shs_change_raw:.2f}%"
        elif shs_enroll2 > 0 : # Growth from 0
            shs_change_raw = 100.0 # Or specific indicator like 'Infinity' if preferred for raw
            shs_change_fmt = "New / Growth from 0"
        else: # Both 0
            shs_change_raw = 0.0
            shs_change_fmt = "0.00%"
            
    kpi_metrics_values = [
        ("Total Enrollment", data1.get('total_enrollment'), data2.get('total_enrollment'), "{:,.0f}"),
        ("Number of Schools", data1.get('num_schools'), data2.get('num_schools'), "{:,.0f}"),
        ("Rate of Change in SHS Enrollment", shs_enroll1, shs_enroll2, shs_change_fmt, shs_change_raw), # Special handling
        ("Overall Gender Parity Index (F/M)", data1.get('gender_parity_index_overall'), data2.get('gender_parity_index_overall'), "{:.2f}")
    ]

    for item in kpi_metrics_values:
        name, v1, v2 = item[0], item[1], item[2]
        fmt = item[3] if len(item) == 4 else None # Standard format
        
        pc_raw, pc_fmt = "N/A", "N/A"
        v1_fmt = "N/A" if v1 is None else (fmt.format(v1) if fmt and isinstance(v1, (int,float)) else str(v1))
        v2_fmt = "N/A" if v2 is None else (fmt.format(v2) if fmt and isinstance(v2, (int,float)) else str(v2))

        if name == "Rate of Change in SHS Enrollment":
            pc_raw = item[4] # Already calculated shs_change_raw
            pc_fmt = item[3] # Already calculated shs_change_fmt
            # v1_fmt and v2_fmt for this specific KPI should show the SHS enrollment numbers
            v1_fmt = format(v1, ',.0f') if isinstance(v1, (int,float)) else "N/A"
            v2_fmt = format(v2, ',.0f') if isinstance(v2, (int,float)) else "N/A"
        elif v1 is not None and v2 is not None and isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if v1 != 0: pc_raw = float(((v2 - v1) / v1 * 100)); pc_fmt = f"{pc_raw:.2f}%"
            elif v2 > 0: pc_raw = 100.0; pc_fmt = "New/Growth from 0"
            else: pc_raw = 0.0; pc_fmt = "0.00%"
        
        results['summary_kpis'].append({
            'metric': str(name), 'year1_value_raw': v1, 'year2_value_raw': v2,
            'year1_value_formatted': v1_fmt, 'year2_value_formatted': v2_fmt,
            'percentage_change_raw': pc_raw, 'percentage_change_formatted': pc_fmt
        })
    
    # Enrollment by Sector & Level
    for key_name in ['enrollment_by_sector', 'enrollment_by_level']:
        d1_sub, d2_sub = data1.get(key_name, {}), data2.get(key_name, {})
        all_cats = sorted(list(set(d1_sub.keys()) | set(d2_sub.keys())))
        cats_to_use = [c for c in all_cats if c != "Unknown"] if key_name == 'enrollment_by_sector' and "Unknown" in all_cats and len(all_cats)>1 else all_cats
        if not cats_to_use and "Unknown" in all_cats: cats_to_use = ["Unknown"]

        if cats_to_use:
            results[key_name] = {
                'categories': [str(c) for c in cats_to_use],
                'year1_values': [d1_sub.get(c, 0) for c in cats_to_use],
                'year2_values': [d2_sub.get(c, 0) for c in cats_to_use]
            }

    # GPI by Level
    gpi_levels = ['Elementary', 'Junior High', 'Senior High']
    d1_gpi, d2_gpi = data1.get('gpi_by_level', {}), data2.get('gpi_by_level', {})
    y1_gpi_vals = [d1_gpi.get(l) for l in gpi_levels]
    y2_gpi_vals = [d2_gpi.get(l) for l in gpi_levels]
    if not (all(v is None for v in y1_gpi_vals) and all(v is None for v in y2_gpi_vals)):
        results['gpi_by_level_comparison'] = {
            'categories': gpi_levels, 'year1_values': y1_gpi_vals, 'year2_values': y2_gpi_vals
        }

    # Gender Distribution
    if data1.get('total_male') is not None or data2.get('total_male') is not None : # Check if any gender data exists
        results['gender_distribution'] = {
            'categories': ['Male', 'Female'],
            'year1_values': [data1.get('total_male', 0), data1.get('total_female', 0)],
            'year2_values': [data2.get('total_male', 0), data2.get('total_female', 0)]
        }

    # Regional Comparison (ordered by absolute change magnitude, then by region name)
    all_regions = sorted([r for r in list(set(data1.get('enrollment_by_region', {}).keys()) | set(data2.get('enrollment_by_region', {}).keys())) if r != "Unknown"])
    reg_data = []
    for r_key in all_regions:
        v1, v2 = data1.get('enrollment_by_region', {}).get(r_key, 0), data2.get('enrollment_by_region', {}).get(r_key, 0)
        abs_c = v2 - v1
        pc_raw, status = 0.0, "Unchanged" # Default for N/A or both zero
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)): # Check if values are numeric
            if v1 != 0: pc_raw = float(((v2 - v1) / v1 * 100))
            elif v2 > 0: status, pc_raw = "New", 100.0 
            elif v1 > 0 and v2 == 0: status, pc_raw = "Ceased", -100.0
        if status == "Unchanged": status = "Growth" if abs_c > 0 else ("Decline" if abs_c < 0 else "Unchanged")
        reg_data.append({'region': str(r_key), 'year1_value': v1, 'year2_value': v2, 'absolute_change': abs_c, 'percentage_change': pc_raw, 'status': str(status)})
    
    # Sort: Primary by status (New, Growth, Decline, Ceased, Unchanged), Secondary by abs change magnitude, Tertiary by region name
    status_order = {"New": 0, "Growth": 1, "Decline": 2, "Ceased": 3, "Unchanged": 4}
    reg_data_sorted = sorted(reg_data, key=lambda x: (status_order.get(x['status'], 5), -abs(x['absolute_change']) if x['status'] in ["Growth", "Decline"] else abs(x['absolute_change']), x['region']))
    
    if reg_data_sorted:
        results['regional_comparison'] = {
            'regions': [r['region'] for r in reg_data_sorted],
            'year1_values': [r['year1_value'] for r in reg_data_sorted],
            'year2_values': [r['year2_value'] for r in reg_data_sorted],
            'absolute_changes': [r['absolute_change'] for r in reg_data_sorted],
            'percentage_changes': [r['percentage_change'] for r in reg_data_sorted],
            'statuses': [r['status'] for r in reg_data_sorted],
            'detailed_data': reg_data_sorted
        }

    # SHS Strand Enrollment
    all_shs_strands = sorted(list(set(data1.get('shs_strand_enrollment', {}).keys()) | set(data2.get('shs_strand_enrollment', {}).keys())))
    if all_shs_strands:
        results['shs_strand_enrollment_comparison'] = {
            'strands': [str(s) for s in all_shs_strands],
            'year1_values': [data1.get('shs_strand_enrollment', {}).get(s, 0) for s in all_shs_strands],
            'year2_values': [data2.get('shs_strand_enrollment', {}).get(s, 0) for s in all_shs_strands]
        }

    # Change in Number of Schools Offering SHS Strands
    schools_strands_data = []
    # Use the same all_shs_strands list
    for strand in all_shs_strands: # Ensure this list is populated if SHS data exists
        s1_count = data1.get('schools_offering_shs_strands', {}).get(strand, 0)
        s2_count = data2.get('schools_offering_shs_strands', {}).get(strand, 0)
        abs_change_schools = s2_count - s1_count
        pct_change_schools_raw = "N/A" # Default to N/A
        if isinstance(s1_count, int) and isinstance(s2_count, int):
            if s1_count != 0:
                pct_change_schools_raw = float((abs_change_schools / s1_count) * 100)
            elif s2_count > 0: 
                pct_change_schools_raw = 100.0 
            elif s1_count == 0 and s2_count == 0:
                pct_change_schools_raw = 0.0
        
        schools_strands_data.append({
            'strand': str(strand), 'year1_count': s1_count, 'year2_count': s2_count,
            'absolute_change': abs_change_schools, 'percentage_change_raw': pct_change_schools_raw
        })
    if schools_strands_data: # Only add if there's data
        results['schools_offering_shs_strands_change'] = {
            'strands': [s['strand'] for s in schools_strands_data],
            'absolute_changes': [s['absolute_change'] for s in schools_strands_data]
            # Add other fields if needed for chart, like year1_counts, year2_counts
        }
        
    # Distribution of Schools by Enrollment Size Category
    dist_categories = ["Very Small (<=50)", "Small (51-200)", "Medium (201-500)", "Large (501-1000)", "Very Large (>1000)", "Unknown"]
    dist1 = data1.get('school_size_distribution', {})
    dist2 = data2.get('school_size_distribution', {})
    if dist1 or dist2: 
        y1c = [dist1.get(cat, 0) for cat in dist_categories]
        y2c = [dist2.get(cat, 0) for cat in dist_categories]
        if sum(y1c) > 0 or sum(y2c) > 0 : # Only create if there are schools in categories
            results['school_size_distribution_comparison'] = {
                'categories': dist_categories, 'year1_counts': y1c, 'year2_counts': y2c
            }
    return results
