import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go # For empty figures or error messages
import pandas as pd
import json
import os
import re

# --- Helper Function to Find Available Datasets ---
def find_heatmap_datasets():
    """
    Scans the 'data_management' folder to find available datasets.
    Assumes datasets are in 'data_management/YYYY-YYYY/filename.csv'.
    Returns a dictionary mapping year string (e.g., "2023-2024") to its file path.
    """
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    except NameError: 
        base_dir = os.getcwd()

    data_management_folder = os.path.join(base_dir, 'data_management')
    datasets = {}

    if not os.path.exists(data_management_folder):
        print(f"HEATMAP DEBUG: Data management folder not found at {data_management_folder}")
        return datasets

    print(f"HEATMAP DEBUG: Scanning for datasets in: {data_management_folder}")
    for year_folder_name in os.listdir(data_management_folder):
        year_folder_path = os.path.join(data_management_folder, year_folder_name)
        if os.path.isdir(year_folder_path):
            if re.match(r'^\d{4}-\d{4}$', year_folder_name):
                for filename in os.listdir(year_folder_path):
                    if filename.lower().endswith('.csv') and year_folder_name in filename:
                        datasets[year_folder_name] = os.path.join(year_folder_path, filename)
                        break 
    if not datasets:
        print("HEATMAP DEBUG: No datasets found. Please check your 'data_management' folder structure and filenames.")
    else:
        print(f"HEATMAP DEBUG: Found datasets: {list(datasets.keys())}") 
    return datasets

# --- Function to Create the Heatmap Dash App ---
def create_heatmap_dash_app(flask_app):
    """
    Creates and configures the Dash application for the Philippine enrollment heatmap.
    """
    heatmap_app = dash.Dash(__name__,
                            server=flask_app,
                            routes_pathname_prefix='/dashheatmap/',
                            external_stylesheets=['/static/style.css']) # Link to your main CSS if needed for consistency

    available_datasets = find_heatmap_datasets()
    available_years = sorted(list(available_datasets.keys()), reverse=True) 

    # Pre-load GeoJSON ADM1_EN values for normalization efficiency
    geojson_adm1_en_values_global = set()
    try:
        try:
            base_dir_init = os.path.dirname(os.path.abspath(__file__))
        except NameError:
            base_dir_init = os.getcwd()
        geojson_path_init = os.path.join(base_dir_init, 'static', 'regions.json')
        with open(geojson_path_init, 'r', encoding='utf-8') as f_init:
            geojson_data_init = json.load(f_init)
            if geojson_data_init and 'features' in geojson_data_init:
                for feature in geojson_data_init['features']:
                    props = feature.get('properties', {})
                    adm1_en_val = props.get('ADM1_EN')
                    if adm1_en_val:
                        geojson_adm1_en_values_global.add(adm1_en_val)
                print(f"HEATMAP INIT DEBUG: Successfully pre-loaded GeoJSON ADM1_EN keys: {sorted(list(geojson_adm1_en_values_global))}")
            else:
                print("HEATMAP INIT ERROR: Pre-loading GeoJSON failed - empty or no features.")
    except Exception as e_init:
        print(f"HEATMAP INIT ERROR: Could not pre-load GeoJSON for ADM1_EN keys: {e_init}")


    # --- Define the layout of the heatmap app ---
    heatmap_app.layout = html.Div([
        html.Div([
            html.Label("Select School Year:", 
                       style={'marginRight': '15px', 
                              'fontWeight': 'bold', 
                              'color': '#00308F', # DepEd Blue
                              'fontSize': '1.1em'}),
            dcc.Dropdown(
                id='year-dropdown-heatmap',
                options=[{'label': year, 'value': year} for year in available_years],
                value=available_years[0] if available_years else None,
                clearable=False,
                style={ # Enhanced dropdown style
                    'width': '300px', 
                    'marginBottom': '20px', 
                    'fontSize': '1em',
                    'border': '1px solid #00308F',
                    'borderRadius': '4px'
                }
            ),
        ], style={ # Style for the dropdown container
            'display': 'flex', 
            'alignItems': 'center', 
            'padding': '15px 20px', 
            'backgroundColor': '#e9ecef', # Light gray background
            'borderRadius': '8px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.05)',
            'marginBottom': '15px'
            }),
        
        dcc.Loading(
            id="loading-heatmap",
            type="circle", # Options: "graph", "cube", "circle", "dot", or "default"
            color="#00308F", # DepEd Blue for spinner
            children=[
                dcc.Graph(
                    id='philippine-heatmap', 
                    style={'height': 'calc(100vh - 150px)', 'width': '100%'} 
                    # Adjust height: 100vh minus approx height of dropdown and padding
                    # This helps ensure the map uses most of the iframe height.
                )
            ]
        )
    ], style={'padding': '10px', 'backgroundColor': '#ffffff'}) # Main app background

    @heatmap_app.callback(
        Output('philippine-heatmap', 'figure'),
        [Input('year-dropdown-heatmap', 'value')]
    )
    def update_heatmap(selected_year):
        print(f"HEATMAP DEBUG: Callback triggered for year: {selected_year}")
        # ... (Error handling for selected_year and data_file_path remains the same) ...
        if not selected_year or selected_year not in available_datasets:
            print(f"HEATMAP DEBUG: No valid year selected ('{selected_year}') or dataset not found.")
            fig = go.Figure()
            fig.add_annotation(text="Please select a valid school year from the dropdown.", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="grey"))
            fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig

        data_file_path = available_datasets[selected_year]

        if not os.path.exists(data_file_path):
            print(f"HEATMAP ERROR: Dataset file not found at {data_file_path}")
            fig = go.Figure()
            fig.add_annotation(text=f"Error: Dataset for {selected_year} not found.", 
                               xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                               font=dict(size=16, color="red"))
            fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig
        
        try:
            df = pd.read_csv(data_file_path)
            print(f"HEATMAP DEBUG: Successfully loaded data for {selected_year} from: {data_file_path}")

            if 'Region' not in df.columns:
                print("HEATMAP ERROR: 'Region' column not found in the dataset.")
                raise ValueError("'Region' column missing in the dataset.")
            
            csv_regions_original = df['Region'].unique().tolist()
            print(f"HEATMAP DEBUG: Unique 'Region' values from CSV (before normalization) for {selected_year}: {csv_regions_original}")

            enrollment_cols = [col for col in df.columns if 'Male' in col or 'Female' in col]
            if not enrollment_cols:
                print("HEATMAP ERROR: No enrollment columns (containing 'Male' or 'Female') found.")
                raise ValueError("No enrollment columns found for aggregation.")
                
            for col in enrollment_cols:
                if col in df.columns: 
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df[enrollment_cols] = df[enrollment_cols].fillna(0)
            df['Total Enrollment'] = df[enrollment_cols].sum(axis=1)
            region_enrollment = df.groupby('Region', as_index=False)['Total Enrollment'].sum()
            
            try:
                base_dir = os.path.dirname(os.path.abspath(__file__)) 
            except NameError:
                base_dir = os.getcwd()
            geojson_path = os.path.join(base_dir, 'static', 'regions.json') 

            try:
                with open(geojson_path, 'r', encoding='utf-8') as f: 
                    geojson_data = json.load(f)
                print("HEATMAP DEBUG: Successfully loaded local GeoJSON from:", geojson_path)
                if not (geojson_data and 'features' in geojson_data and len(geojson_data['features']) > 0):
                    print("HEATMAP ERROR: GeoJSON data is empty or has no features.")
                    raise ValueError("GeoJSON data is invalid.")
            except FileNotFoundError:
                print(f"HEATMAP ERROR: GeoJSON file not found at {geojson_path}.")
                raise ValueError("GeoJSON map file 'regions.json' missing from 'static' folder.")
            except json.JSONDecodeError:
                print(f"HEATMAP ERROR: Could not decode GeoJSON file at {geojson_path}.")
                raise ValueError("GeoJSON file is corrupted or not valid JSON.")

            # --- Normalize Region Names ---
            def normalize_region_name_for_map(name_from_csv):
                name_upper = str(name_from_csv).upper().strip()
                
                # This mapping MUST ensure the VALUE is an EXACT match to an ADM1_EN in your GeoJSON
                mapping = {
                    "NCR": "National Capital Region",
                    "CAR": "Cordillera Administrative Region",
                    "REGION I": "Region I", 
                    "REGION II": "Region II",
                    "REGION III": "Region III", 
                    "REGION IV-A": "Region IV-A", 
                    "CALABARZON": "Region IV-A",    
                    "MIMAROPA": "Region IV-B",      
                    "REGION IV-B": "Region IV-B",
                    "REGION IV-B - MIMAROPA": "Region IV-B", 
                    "REGION V": "Region V",
                    "BICOL REGION" : "Region V",
                    "REGION VI": "Region VI",
                    "WESTERN VISAYAS": "Region VI",
                    "REGION VII": "Region VII",
                    "CENTRAL VISAYAS": "Region VII",
                    "REGION VIII": "Region VIII",
                    "EASTERN VISAYAS": "Region VIII",
                    "REGION IX": "Region IX",
                    "ZAMBOANGA PENINSULA": "Region IX",
                    "REGION X": "Region X",
                    "NORTHERN MINDANAO": "Region X",
                    "REGION XI": "Region XI",
                    "DAVAO REGION": "Region XI",
                    "REGION XII": "Region XII",
                    "SOCCSKSARGEN": "Region XII",
                    "CARAGA": "Region XIII", 
                    "REGION XIII": "Region XIII",
                    "ARMM": "Autonomous Region in Muslim Mindanao", 
                    "BARMM": "Bangsamoro Autonomous Region in Muslim Mindanao" 
                }
                
                normalized_value = mapping.get(name_upper)
                
                if normalized_value:
                    return normalized_value
                else:
                    # Fallback: if the uppercased CSV name is directly in the GeoJSON ADM1_EN list (case-insensitive check)
                    for geo_key in geojson_adm1_en_values_global: # Use the pre-loaded set
                        if name_upper == geo_key.upper():
                            print(f"HEATMAP DEBUG: Normalizing '{name_from_csv}' to '{geo_key}' (direct case-insensitive match with GeoJSON ADM1_EN)")
                            return geo_key
                    
                    print(f"HEATMAP WARNING: Region name '{name_from_csv}' (normalized to '{name_upper}') could not be mapped. This region might not color correctly.")
                    return name_upper 

            region_enrollment['RegionMapKey'] = region_enrollment['Region'].apply(normalize_region_name_for_map)
            
            valid_geojson_keys = geojson_adm1_en_values_global 
            region_enrollment_filtered = region_enrollment[region_enrollment['RegionMapKey'].isin(valid_geojson_keys)]

            if region_enrollment_filtered.empty and not region_enrollment.empty:
                print("HEATMAP WARNING: After normalization, no regions in the CSV data matched any ADM1_EN in the GeoJSON. All regions might appear uncolored.")
            
            print(f"HEATMAP DEBUG: Unique 'RegionMapKey' values passed to map (after normalization & filtering): {region_enrollment_filtered['RegionMapKey'].unique().tolist()}")
            print(f"HEATMAP DEBUG: Data for choropleth (first 5 rows, filtered):\n{region_enrollment_filtered[['Region', 'RegionMapKey', 'Total Enrollment']].head()}")

            fig = px.choropleth_mapbox(region_enrollment_filtered, 
                                       geojson=geojson_data,
                                       locations='RegionMapKey', 
                                       featureidkey="properties.ADM1_EN", 
                                       color='Total Enrollment',
                                       color_continuous_scale="YlOrRd", 
                                       range_color=(0, region_enrollment_filtered['Total Enrollment'].max() if not region_enrollment_filtered.empty and region_enrollment_filtered['Total Enrollment'].max() > 0 else 1),
                                       mapbox_style="carto-positron", 
                                       zoom=4.7,
                                       center = {"lat": 12.8797, "lon": 121.7740}, 
                                       opacity=0.75, # Slightly more opaque
                                       labels={'Total Enrollment':f'Enrollees ({selected_year})', 'RegionMapKey': 'Region'},
                                       hover_name='RegionMapKey', # Show the matched GeoJSON key on hover for debugging
                                       hover_data={'Region': True, 'Total Enrollment': True, 'RegionMapKey': False} # Show original CSV region and total
                                      )
            
            fig.update_layout(
                margin={"r":0,"t":50,"l":0,"b":0}, # Increased top margin for title
                mapbox_accesstoken=None, 
                title_text=f'<b>Regional Enrollment Heatmap - SY {selected_year}</b>', # Bold title
                title_x=0.5, 
                title_font_size=20,
                title_font_family="Montserrat, sans-serif",
                coloraxis_colorbar=dict(
                    title="Total Enrollees",
                    thicknessmode="pixels", thickness=18, # Thicker color bar
                    lenmode="fraction", len=0.75, 
                    yanchor="middle", y=0.5,
                    tickfont_size=10,
                    title_font_size=12,
                    bgcolor='rgba(255,255,255,0.7)', # Semi-transparent background
                    bordercolor='#cccccc',
                    borderwidth=1
                ),
                paper_bgcolor='rgba(0,0,0,0)', # Transparent background for the figure itself
                plot_bgcolor='rgba(0,0,0,0)'
            )
            if region_enrollment_filtered.empty and not region_enrollment.empty:
                 fig.add_annotation(text="No data to display on map after region name matching.",
                                   xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
                                   font=dict(size=16, color="orange"))

            print(f"HEATMAP DEBUG: Heatmap figure for {selected_year} created successfully.")
            return fig

        except ValueError as ve: 
            print(f"HEATMAP ERROR: ValueError during heatmap generation for {selected_year}: {ve}")
            fig = go.Figure()
            fig.add_annotation(text=f"Data Error: {str(ve)}", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="orange"))
            fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig
        except Exception as e:
            print(f"HEATMAP ERROR: Unexpected error generating heatmap for {selected_year}: {e}")
            import traceback
            traceback.print_exc()
            fig = go.Figure()
            fig.add_annotation(text=f"An unexpected error occurred. Please check console.", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="red"))
            fig.update_layout(xaxis={'visible': False}, yaxis={'visible': False}, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            return fig

    return heatmap_app