# report.py

from dash import Dash, html, dcc, Input, Output, State, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
# Ensure data_config parts are still relevant or adapt
from data_config import fetch_enrollment_records_from_csv, fetch_summary_data_from_csv # get_dataset_path will be replaced
import io
import base64
import re
import numpy as np
from flask import session # May not be needed if all state is in Dash
import os # Added

# Define DATA_MANAGEMENT_FOLDER (mirroring app.py or passed as arg)
# For simplicity, defining it here. In a larger app, this might come from a central config.
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_MANAGEMENT_FOLDER = os.path.join(BASE_DIR, 'data_management')


GENDER_COLORS = {'Male': '#1f77b4', 'Female': '#e377c2', 'Unknown': '#888'}
QUALITATIVE_COLOR_SEQUENCE = px.colors.qualitative.Pastel
PLOT_TEMPLATE = "plotly_white"

# Helper function to find available datasets (adapted from app.py)
def find_available_datasets_for_report():
    datasets_paths = {}
    years = []
    if not os.path.exists(DATA_MANAGEMENT_FOLDER):
        print(f"Warning: Data management folder not found at {DATA_MANAGEMENT_FOLDER}")
        return {}, []

    for year_dir in os.listdir(DATA_MANAGEMENT_FOLDER):
        year_path_full = os.path.join(DATA_MANAGEMENT_FOLDER, year_dir)
        if os.path.isdir(year_path_full):
            if "-" in year_dir: # Basic check for school year format
                for filename in os.listdir(year_path_full):
                    if filename.lower().endswith(".csv"):
                        full_path = os.path.join(year_path_full, filename)
                        datasets_paths[year_dir] = full_path
                        years.append(year_dir)
                        break # Found first CSV for this year
    years.sort(reverse=True) # Latest year first
    return datasets_paths, years

def create_dash_app_report(flask_app):
    dash_app_report = Dash(
        __name__,
        server=flask_app,
        routes_pathname_prefix="/dashreport/",
        external_stylesheets=['/static/style.css'], # Assuming style.css is in static folder
        suppress_callback_exceptions=True,
        serve_locally=True # Ensure assets are served if not using CDN
    )

    datasets_paths, available_years = find_available_datasets_for_report()
    default_year = available_years[0] if available_years else None
    default_file_path = datasets_paths.get(default_year) if default_year else None

    # --- DASH LAYOUT ---
    dash_app_report.layout = html.Div([
        dcc.Store(id='selected-school-year-path-store', data=default_file_path),
        dcc.Store(id='main-data-store'), # Stores df_all as JSON
        dcc.Store(id='enrollment-cols-store'), # Stores list of enrollment columns
        dcc.Store(id='filter-options-store'), # Stores dict of options for dropdowns

        html.H1("📊 School Enrollment Dashboard", style={
            "textAlign": "center", "marginBottom": "20px", "color": "#333", "fontSize": "2rem"
        }),

        # School Year Selector
        html.Div([
            html.Label("📅 School Year", style={'margin-right': '10px'}),
            dcc.Dropdown(
                id='school-year-filter',
                options=[{'label': year, 'value': year} for year in available_years],
                value=default_year, # Default to the latest year
                clearable=False,
                style={'width': '250px'}
            )
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),

        html.Div([ # Filters Container
            html.Div([
                html.Label("🔍 Region"),
                dcc.Dropdown(id='region-filter', placeholder="Select Region")
            ], className="filter-item"),
            html.Div([
                html.Label("📍 Division"),
                dcc.Dropdown(id='division-filter', placeholder="Select Division", disabled=True)
            ], className="filter-item"),
            html.Div([
                html.Label("🎓 Grade Level"),
                dcc.Dropdown(id='grade-filter', placeholder="Select Grade Level")
            ], className="filter-item"),
            html.Div([
                html.Label("🏫 Sector"),
                dcc.RadioItems(id='sector-filter', inline=True)
            ], className="filter-item"),
            html.Div([
                html.Label("🔑 School (BEIS ID)"),
                dcc.Dropdown(id='beis-id-filter', placeholder="Select School (Optional)")
            ], className="filter-item filter-item-wide"),
        ], className="filters-container"),

        html.Div([
            html.Button("Reset Filters", id="reset-button", n_clicks=0, className="reset-button"),
            html.Button("⬇ Download Data", id="btn-download", n_clicks=0, className="download-button"),
            html.A("Comparison", href="/compare", className="compare-page-button") # Ensure this class is styled
        ], style={"display": "flex", "justifyContent": "center", "margin": "15px 0", "gap": "16px", "flexWrap": "wrap"}),
        html.Hr(),
        dcc.Loading(id="loading-kpi", type="circle", children=html.Div(id='kpi-cards', className="kpi-cards-container")),
        html.Hr(),
        html.H2("Overview Analysis", style={"textAlign": "center", "marginBottom": "15px", "color": "#555"}),
        html.Div(className="row", children=[
            dcc.Loading(type="circle", children=dcc.Graph(id='region-enrollment-bar', className="graph-item-half")),
            dcc.Loading(type="circle", children=dcc.Graph(id='grade-gender-parity-bar', className="graph-item-half")),
        ]),
        # ... (other graph divs remain the same) ...
        html.Div(className="row", children=[
            dcc.Loading(type="circle", children=dcc.Graph(id='sector-distribution', className="graph-item-half")),
            dcc.Loading(type="circle", children=dcc.Graph(id='education-stage-distribution', className="graph-item-half")),
        ]),

        html.Div(className="row", children=[
            dcc.Loading(type="circle", children=dcc.Graph(id='shs-strand-gender-combined-bar', className="graph-item-half")),
            dcc.Loading(type="circle", children=dcc.Graph(id='strand-enrollment-trend-line', className="graph-item-half")),
        ]),

        html.Div(className="row", children=[
            dcc.Loading(type="circle", children=dcc.Graph(id='top-populated-schools-bar', className="graph-item-half")),
            dcc.Loading(type="circle", children=dcc.Graph(id='schools-offering-strand-treemap', className="graph-item-half")),
        ]),

        html.Hr(style={"marginTop": "25px", "marginBottom": "15px"}),
        html.H2("⚠️ Watchlist: Least Populated Schools", style={"textAlign": "center", "marginBottom": "12px", "color": "#E69F00"}),
        dcc.Loading(id="loading-table", type="circle", children=html.Div(id='flagged-schools-table', className="table-container")),
        html.Br(),
        dcc.Download(id="download-data")

    ], className="main-container", style={"backgroundColor": "#f0f2f5", "padding": "40px", "maxWidth": "1200px", "margin": "20px auto"})

    # --- CALLBACKS ---

    # Callback 1: Update selected school year path
    @dash_app_report.callback(
        Output('selected-school-year-path-store', 'data'),
        Input('school-year-filter', 'value')
    )
    def update_selected_school_year_path(selected_year):
        if selected_year and selected_year in datasets_paths:
            return datasets_paths[selected_year]
        return None

    # Callback 2: Load data based on selected school year and initialize filters
    @dash_app_report.callback(
        Output('main-data-store', 'data'),
        Output('enrollment-cols-store', 'data'),
        Output('filter-options-store', 'data'),
        Output('region-filter', 'options'), Output('region-filter', 'value'),
        Output('division-filter', 'options'), Output('division-filter', 'value'), # Division also reset
        Output('grade-filter', 'options'), Output('grade-filter', 'value'),
        Output('sector-filter', 'options'), Output('sector-filter', 'value'),
        Output('beis-id-filter', 'options'), Output('beis-id-filter', 'value'),
        Input('selected-school-year-path-store', 'data'),
        prevent_initial_call=False # Run on load with default year
    )
    def load_data_and_initialize_filters(file_path):
        if not file_path or not os.path.exists(file_path):
            # Return empty/default states if no file path or file doesn't exist
            empty_options = [{'label': 'No Data', 'value': ''}]
            empty_df_json = pd.DataFrame().to_json(date_format='iso', orient='split')
            return empty_df_json, [], {}, empty_options, None, empty_options, None, empty_options, None, empty_options, None, empty_options, None

        try:
            df = pd.read_csv(file_path) # Using read_csv directly
            # Basic preprocessing (adapt from original)
            enrollment_patterns = ['K ', 'G1 ', 'G2 ', 'G3 ', 'G4 ', 'G5 ', 'G6 ', 'G7 ', 'G8 ', 'G9 ', 'G10 ', 'G11 ', 'G12 ', 'Elem NG ', 'JHS NG ']
            enrollment_cols = [
                col for col in df.columns
                if any(col.startswith(pattern) for pattern in enrollment_patterns)
                and (' Male' in col or ' Female' in col)
            ]
            for col in enrollment_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df[enrollment_cols] = df[enrollment_cols].fillna(0).astype(int)

            id_cat_cols = ['BEIS School ID', 'Region', 'Division', 'Sector', 'School Type', 'School Subclassification', 'Municipality', 'Legislative District', 'School Name']
            for col in id_cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype(str).fillna('Unknown')
                elif col == 'BEIS School ID': # Ensure BEIS School ID always exists
                    df[col] = 'Unknown'


            # Prepare filter options
            regions_opts = []
            divisions_by_region_dict = {}
            all_divisions_opts = []
            beis_opts = []
            sector_opts = []
            grades_opts = [{'label': g, 'value': g} for g in ['K'] + [f'G{i}' for i in range(1, 11)] + ['G11', 'G12']]


            if "Region" in df.columns:
                regions_list = sorted([r for r in df["Region"].unique() if r != 'Unknown'])
                regions_opts = [{'label': r, 'value': r} for r in regions_list]

            if "Region" in df.columns and "Division" in df.columns:
                df_filtered_for_div = df[(df["Region"] != 'Unknown') & (df["Division"] != 'Unknown')]
                if not df_filtered_for_div.empty:
                    divisions_by_region_dict = df_filtered_for_div.groupby("Region")["Division"].unique().apply(lambda x: sorted(list(x))).to_dict()
                    all_divs_list = sorted(df_filtered_for_div["Division"].unique())
                    all_divisions_opts = [{'label': d, 'value': d} for d in all_divs_list]
                else: # Fallback if no valid region/division pairs
                    all_divisions_opts = [{'label': d, 'value': d} for d in sorted(df["Division"].unique()) if d != 'Unknown']


            if "BEIS School ID" in df.columns and "School Name" in df.columns:
                unique_schools_df = df[["BEIS School ID", "School Name"]].drop_duplicates(subset=['BEIS School ID'])
                if not unique_schools_df.empty:
                    all_beis_items = sorted(unique_schools_df.set_index("BEIS School ID")["School Name"].to_dict().items())
                    beis_opts = [{'label': f"{id_} - {name[:50]}{'...' if len(name)>50 else ''}", 'value': id_} for id_, name in all_beis_items if id_ != 'Unknown']

            if "Sector" in df.columns:
                sector_list = sorted([s for s in df["Sector"].unique() if s != 'Unknown'])
                sector_opts = [{'label': s, 'value': s} for s in sector_list]

            filter_options_data = {
                'divisions_by_region': divisions_by_region_dict,
                'all_divisions_opts': all_divisions_opts, # For when no region is selected
                'all_beis_opts': beis_opts # For when no region/division is selected
            }

            return (df.to_json(date_format='iso', orient='split'),
                    enrollment_cols,
                    filter_options_data,
                    regions_opts, None, # region options, default value
                    all_divisions_opts, None, # division options, default value
                    grades_opts, None, # grade options, default value
                    sector_opts, None, # sector options, default value
                    beis_opts, None)   # beis_id options, default value

        except Exception as e:
            print(f"Error loading data or preparing filters for {file_path}: {e}")
            empty_options = [{'label': 'Error Loading', 'value': ''}]
            empty_df_json = pd.DataFrame().to_json(date_format='iso', orient='split')
            return empty_df_json, [], {}, empty_options, None, empty_options, None, empty_options, None, empty_options, None, empty_options, None


    # Callback to update Division dropdown based on Region
    @dash_app_report.callback(
        Output('division-filter', 'options', allow_duplicate=True),
        Output('division-filter', 'disabled', allow_duplicate=True),
        Input('region-filter', 'value'),
        State('filter-options-store', 'data'), # Get pre-calculated options
        prevent_initial_call=True
    )
    def update_divisions_dropdown(selected_region, filter_options_data):
        if not filter_options_data: # Store not ready
            return [], True

        divisions_by_region = filter_options_data.get('divisions_by_region', {})
        all_divisions_opts = filter_options_data.get('all_divisions_opts', [])

        if selected_region and selected_region != 'Unknown' and selected_region in divisions_by_region:
            options = [{'label': d, 'value': d} for d in divisions_by_region[selected_region]]
            return options, False
        return all_divisions_opts, False # Enable and show all divisions if no specific region or data issue


    # Callback to update BEIS School ID options
    @dash_app_report.callback(
        Output('beis-id-filter', 'options', allow_duplicate=True),
        Input('region-filter', 'value'),
        Input('division-filter', 'value'),
        State('main-data-store', 'data'), # Get the full dataframe for filtering
        State('filter-options-store', 'data'), # Get all_beis_opts for fallback
        prevent_initial_call=True
    )
    def update_beis_ids_dropdown(selected_region, selected_division, df_json, filter_options_data):
        if not df_json or not filter_options_data:
            return filter_options_data.get('all_beis_opts', []) if filter_options_data else []

        df_all = pd.read_json(df_json, orient='split')
        all_beis_opts_fallback = filter_options_data.get('all_beis_opts', [])

        if df_all.empty:
            return all_beis_opts_fallback

        filtered_schools_df = df_all.copy()
        if selected_region and selected_region != 'Unknown':
            filtered_schools_df = filtered_schools_df[filtered_schools_df['Region'] == selected_region]
        if selected_division and selected_division != 'Unknown':
            filtered_schools_df = filtered_schools_df[filtered_schools_df['Division'] == selected_division]

        beis_ids_with_names_filtered = []
        if 'BEIS School ID' in filtered_schools_df.columns and 'School Name' in filtered_schools_df.columns and not filtered_schools_df.empty:
            unique_filtered_schools = filtered_schools_df[["BEIS School ID", "School Name"]].drop_duplicates(subset=['BEIS School ID'])
            beis_items = sorted(unique_filtered_schools.set_index("BEIS School ID")["School Name"].to_dict().items())
            beis_ids_with_names_filtered = [{'label': f"{id_} - {name[:50]}{'...' if len(name)>50 else ''}", 'value': id_} for id_, name in beis_items if id_ != 'Unknown']

        return beis_ids_with_names_filtered if beis_ids_with_names_filtered else all_beis_opts_fallback

    # Reset Filters Callback
    @dash_app_report.callback(
        Output('region-filter', 'value', allow_duplicate=True),
        Output('division-filter', 'value', allow_duplicate=True),
        Output('grade-filter', 'value', allow_duplicate=True),
        Output('sector-filter', 'value', allow_duplicate=True),
        Output('beis-id-filter', 'value', allow_duplicate=True),
        # Output('school-year-filter', 'value'), # Optionally reset school year too, or keep it
        Input('reset-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def reset_filters(n_clicks):
        return None, None, None, None, None #, default_year # if resetting school year


    # --- Main Update Callback for visualizations ---
    @dash_app_report.callback(
        Output('kpi-cards', 'children'),
        Output('region-enrollment-bar', 'figure'),
        Output('grade-gender-parity-bar', 'figure'),
        Output('sector-distribution', 'figure'),
        Output('education-stage-distribution', 'figure'),
        Output('shs-strand-gender-combined-bar', 'figure'),
        Output('top-populated-schools-bar', 'figure'),
        Output('strand-enrollment-trend-line', 'figure'),
        Output('schools-offering-strand-treemap', 'figure'),
        Output('flagged-schools-table', 'children'),
        Input('main-data-store', 'data'),
        Input('enrollment-cols-store', 'data'),
        Input('region-filter', 'value'),
        Input('division-filter', 'value'),
        Input('grade-filter', 'value'),
        Input('sector-filter', 'value'),
        Input('beis-id-filter', 'value'),
    )
    def update_dashboard(df_json, enrollment_cols, selected_region, selected_division, selected_grade, selected_sector, selected_beis_id):
        # Placeholder Figure Function
        def create_placeholder_figure(title_text):
            fig = go.Figure()
            fig.add_annotation(text="No data or waiting for selection...", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=12, color="#888"))
            fig.update_layout(title=title_text, title_x=0.5, xaxis={'visible': False}, yaxis={'visible': False}, template=PLOT_TEMPLATE, title_font_size=14, height=400, width=600)
            return fig

        empty_kpis = html.Div([
            html.Div([html.H3("Total Enrolled", className="kpi-title"), html.H1("N/A", className="kpi-value")], className="kpi-card"),
            html.Div([html.H3("Male vs Female", className="kpi-title"), html.P("N/A", className="kpi-value")], className="kpi-card"),
            html.Div([html.H3("Number of Schools", className="kpi-title"), html.H1("N/A", className="kpi-value")], className="kpi-card"),
            html.Div([html.H3("Most Populated Grade", className="kpi-title"), html.H1("N/A", className="kpi-value")], className="kpi-card"),
        ], className="kpi-cards-container")
        empty_table = html.P("No data for watchlist.", style={"textAlign": "center"})


        if not df_json or not enrollment_cols:
            figs = [create_placeholder_figure(title) for title in [
                "Total Enrollment by Region", "Enrollment by Grade Level and Gender",
                "Enrollment Distribution by Sector", "Enrollment Distribution by Education Stage",
                "SHS Enrollment by Strand and Gender", "Top 10 Most Populated Schools",
                "SHS Strand Enrollment: G11 vs G12", "Number of Schools Offering Each SHS Strand"
            ]]
            return (empty_kpis, *figs, empty_table)

        df_all = pd.read_json(df_json, orient='split')
        if df_all.empty: # Double check after parsing JSON
            figs = [create_placeholder_figure(title) for title in [
                "Total Enrollment by Region", "Enrollment by Grade Level and Gender",
                "Enrollment Distribution by Sector", "Enrollment Distribution by Education Stage",
                "SHS Enrollment by Strand and Gender", "Top 10 Most Populated Schools",
                "SHS Strand Enrollment: G11 vs G12", "Number of Schools Offering Each SHS Strand"
            ]]
            return (empty_kpis, *figs, empty_table)


        # --- Filtering Logic --- (Copied and adapted from original, using df_all and enrollment_cols from inputs)
        filtered_df = df_all.copy()
        query_parts = []
        if selected_region and selected_region != 'Unknown': query_parts.append(f"Region == '{selected_region}'")
        if selected_division and selected_division != 'Unknown': query_parts.append(f"Division == '{selected_division}'")
        if selected_sector and selected_sector != 'Unknown': query_parts.append(f"Sector == '{selected_sector}'")
        if selected_beis_id and selected_beis_id != 'Unknown': query_parts.append(f"`BEIS School ID` == '{str(selected_beis_id)}'")

        if query_parts:
            try:
                filtered_df = filtered_df.query(" and ".join(query_parts))
            except Exception as e:
                print(f"Error querying DataFrame: {e}") # Handle potential query errors
                # Fallback to unfiltered or less filtered data if query fails.
                # For now, just print and continue with potentially un-fully-filtered df.


        filtered_df_base_after_geo_sector_id = filtered_df.copy() # This was how it was done before grade filtering

        current_enrollment_cols = list(enrollment_cols) # Use the passed enrollment_cols
        if selected_grade:
            grade_columns_to_keep = []
            for col in enrollment_cols: # Iterate over the original full list of enrollment_cols
                if col in filtered_df.columns: # Check if col exists in the possibly filtered_df
                    if col.startswith(selected_grade + " ") or (selected_grade == 'K' and col.startswith('K ')):
                        grade_columns_to_keep.append(col)
            current_enrollment_cols = grade_columns_to_keep


        numeric_enrollment_cols_filtered = [
            col for col in current_enrollment_cols # Use current_enrollment_cols derived from selected_grade
            if col in filtered_df.columns and pd.api.types.is_numeric_dtype(filtered_df[col])
        ]
        all_numeric_enrollment_cols_in_base = [
             col for col in enrollment_cols # Use the full enrollment_cols list from store
             if col in filtered_df_base_after_geo_sector_id.columns and pd.api.types.is_numeric_dtype(filtered_df_base_after_geo_sector_id[col])
        ]


        # KPIs Calculation
        total_enrollments = filtered_df[numeric_enrollment_cols_filtered].sum().sum() if numeric_enrollment_cols_filtered and not filtered_df.empty else 0
        male_enrollments = filtered_df[[col for col in numeric_enrollment_cols_filtered if 'Male' in col]].sum().sum() if numeric_enrollment_cols_filtered and not filtered_df.empty else 0
        female_enrollments = filtered_df[[col for col in numeric_enrollment_cols_filtered if 'Female' in col]].sum().sum() if numeric_enrollment_cols_filtered and not filtered_df.empty else 0

        number_of_schools = 0
        if 'BEIS School ID' in filtered_df_base_after_geo_sector_id.columns and not filtered_df_base_after_geo_sector_id.empty:
            number_of_schools = filtered_df_base_after_geo_sector_id['BEIS School ID'].nunique()

        most_populated_grade = "N/A"
        if not filtered_df_base_after_geo_sector_id.empty and all_numeric_enrollment_cols_in_base:
            grade_enrollment_base = {}
            for col in all_numeric_enrollment_cols_in_base:
                match = re.match(r'(K|G\d{1,2}|Elem NG|JHS NG)\b', col)
                if match:
                    grade = match.group(1)
                    grade_enrollment_base[grade] = grade_enrollment_base.get(grade, 0) + filtered_df_base_after_geo_sector_id[col].sum()
            ordered_grades_keys = ['K'] + [f'G{i}' for i in range(1, 13)] + ['Elem NG', 'JHS NG'] # Max G12
            present_ordered_grades = [g for g in ordered_grades_keys if g in grade_enrollment_base]

            if grade_enrollment_base:
                 max_enrollment = -1
                 # Initialize most_populated_grade before loop
                 most_populated_grade_candidate = "N/A"
                 for grade_key in present_ordered_grades: # Iterate in defined order
                     if grade_enrollment_base.get(grade_key, 0) > max_enrollment:
                         max_enrollment = grade_enrollment_base[grade_key]
                         most_populated_grade_candidate = grade_key
                 most_populated_grade = most_populated_grade_candidate


        kpis = html.Div([
            html.Div([html.H3("Total Enrolled", className="kpi-title"), html.H1(f"{total_enrollments:,}", className="kpi-value")], className="kpi-card"),
            html.Div([html.H3("Male vs Female", className="kpi-title"), html.P(f"{male_enrollments:,} ♂ | {female_enrollments:,} ♀", className="kpi-value")], className="kpi-card"),
            html.Div([html.H3("Number of Schools", className="kpi-title"), html.H1(f"{number_of_schools:,}", className="kpi-value")], className="kpi-card"),
            html.Div([html.H3("Most Populated Grade", className="kpi-title"), html.H1(most_populated_grade, className="kpi-value")], className="kpi-card"),
        ], className="kpi-cards-container")


        # --- Generate Overview Figures ---
        # IMPORTANT: The following plotting logic is largely copied.
        # It needs to correctly use df_all (for school-year wide context for some plots)
        # and filtered_df (for context-specific plots).
        # The original code sometimes used df_all for region plots, and filtered_df for others.
        # This distinction needs to be maintained or clarified based on the desired behavior with school year changes.
        # For now, I will assume plots that were using the global df_all should now use the
        # df_all loaded for the specific school year (which is now the df_all passed into this callback).

        # 1. Region Enrollment Bar
        fig_region_bar = create_placeholder_figure("Total Enrollment by Region")
        # The original logic for this plot used the global df_all unless a region was selected,
        # then it used filtered_df_base_after_geo_sector_id.
        # With the new structure, df_all IS the data for the selected school year.
        # If a region is selected, filtered_df_base_after_geo_sector_id is based on that region within the school year.
        # Let's use df_all (school year data) if no region selected, else the region-filtered data.
        region_df_for_chart = df_all if not selected_region else filtered_df_base_after_geo_sector_id
        all_numeric_cols_region = [col for col in enrollment_cols if col in region_df_for_chart.columns and pd.api.types.is_numeric_dtype(region_df_for_chart[col])]

        if 'Region' in region_df_for_chart.columns and not region_df_for_chart.empty and all_numeric_cols_region:
             region_enrollment = region_df_for_chart.groupby("Region")[all_numeric_cols_region].sum().sum(axis=1).reset_index(name='Total Enrollment')
             region_enrollment = region_enrollment[(region_enrollment['Total Enrollment'] > 0) & (region_enrollment['Region'] != 'Unknown')].sort_values('Total Enrollment', ascending=False)
             if not region_enrollment.empty:
                 fig_region_bar = px.bar(region_enrollment, x="Region", y="Total Enrollment", title="Total Enrollment by Region", labels={'Total Enrollment': 'Enrollment Count', 'Region': 'Region'}, template=PLOT_TEMPLATE, color_discrete_sequence=QUALITATIVE_COLOR_SEQUENCE)
                 fig_region_bar.update_layout(title_font_size=14, title_x=0.5, xaxis={'categoryorder':'total descending'}, height=400, width=600)

        # 2. Grade Gender Parity Bar Chart
        fig_grade_gender_parity = create_placeholder_figure("Enrollment by Grade Level and Gender")
        # This plot uses numeric_enrollment_cols_filtered which is based on selected_grade and other filters.
        grade_parity_cols_candidate = [col for col in numeric_enrollment_cols_filtered if re.match(r'(K|G\d{1,2})\b', col)]

        if grade_parity_cols_candidate and not filtered_df.empty:
             # Ensure BEIS School ID exists, or add a temporary one for melting if not present in filtered_df
             temp_id_added = False
             if 'BEIS School ID' not in filtered_df.columns:
                 # This case should ideally not happen if 'BEIS School ID' is ensured during load
                 # For robustness, if it's missing, we might skip or handle carefully.
                 # For melt, an ID var is useful. Let's assume it's there.
                 pass

             melt_df_parity = filtered_df[['BEIS School ID'] + grade_parity_cols_candidate] if 'BEIS School ID' in filtered_df.columns else filtered_df[grade_parity_cols_candidate]
             melt_id_vars_parity = ['BEIS School ID'] if 'BEIS School ID' in filtered_df.columns else []

             if not melt_id_vars_parity and not melt_df_parity.empty: # If no ID and df not empty
                  melt_df_parity['_temp_id_parity'] = range(len(melt_df_parity))
                  melt_id_vars_parity = ['_temp_id_parity']
                  temp_id_added = True

             if not melt_df_parity.empty: # Check before melting
                melted_parity = pd.melt(melt_df_parity, id_vars=melt_id_vars_parity, value_vars=grade_parity_cols_candidate, var_name="GradeGender", value_name="Count")

                if temp_id_added and '_temp_id_parity' in melted_parity.columns:
                    melted_parity = melted_parity.drop(columns=['_temp_id_parity'])

                melted_parity = melted_parity[melted_parity['Count'] > 0]

                if not melted_parity.empty:
                    melted_parity["Grade"] = melted_parity["GradeGender"].str.extract(r'(K|G\d{1,2})\b').fillna('Unknown Grade')
                    melted_parity["Gender"] = melted_parity["GradeGender"].str.extract(r'(Male|Female)$').fillna('Unknown')

                    parity_data = melted_parity.groupby(["Grade", "Gender"], observed=False)["Count"].sum().reset_index()
                    all_possible_grades_chart = ['K'] + [f'G{i}' for i in range(1, 13)] # Max G12
                    parity_data['Grade'] = pd.Categorical(parity_data['Grade'], categories=all_possible_grades_chart, ordered=True)
                    parity_data = parity_data.sort_values('Grade')

                    fig_grade_gender_parity = px.bar(
                        parity_data, x="Grade", y="Count", color="Gender", barmode="group",
                        title="Enrollment by Grade Level and Gender",
                        labels={'Count': 'Students', 'Grade': 'Grade Level', 'Gender': 'Gender'},
                        category_orders={"Grade": all_possible_grades_chart}, template=PLOT_TEMPLATE, color_discrete_map=GENDER_COLORS
                    )
                    fig_grade_gender_parity.update_layout(title_font_size=14, legend_title_text='Gender', title_x=0.5, height=400, width=600)

        # 3. Sector Distribution Pie Chart
        fig_sector = create_placeholder_figure('Enrollment Distribution by Sector')
        if 'Sector' in filtered_df.columns and not filtered_df.empty and numeric_enrollment_cols_filtered:
             sector_enrollment = filtered_df.groupby("Sector")[numeric_enrollment_cols_filtered].sum().sum(axis=1).reset_index(name='Total Enrollment')
             sector_enrollment = sector_enrollment[(sector_enrollment['Total Enrollment'] > 0) & (sector_enrollment['Sector'] != 'Unknown')]
             if not sector_enrollment.empty:
                  fig_sector = px.pie(
                      sector_enrollment, names='Sector', values='Total Enrollment', title='Enrollment Distribution by Sector (Filtered)',
                      hole=0.4, template=PLOT_TEMPLATE, color_discrete_sequence=QUALITATIVE_COLOR_SEQUENCE)
                  fig_sector.update_traces(textposition='outside', textinfo='percent+label', pull=[0.05]*len(sector_enrollment))
                  fig_sector.update_layout(title_font_size=14, legend_title_text='Sector', title_x=0.5, uniformtext_minsize=10, uniformtext_mode='hide', height=400, width=600)

        # 4. Education Stage Distribution Pie Chart
        fig_education_stage = create_placeholder_figure('Enrollment Distribution by Education Stage')
        if not filtered_df.empty and numeric_enrollment_cols_filtered:
             elem_cols = [col for col in numeric_enrollment_cols_filtered if any(col.startswith(g + ' ') for g in ['K'] + [f'G{i}' for i in range(1, 7)]) or 'Elem NG' in col]
             jhs_cols = [col for col in numeric_enrollment_cols_filtered if any(col.startswith(g + ' ') for g in [f'G{i}' for i in range(7, 11)]) or 'JHS NG' in col]
             shs_cols = [col for col in numeric_enrollment_cols_filtered if any(col.startswith(g + ' ') for g in ['G11', 'G12'])]

             elementary = filtered_df[elem_cols].sum().sum() if elem_cols and not filtered_df[elem_cols].empty else 0
             junior_high = filtered_df[jhs_cols].sum().sum() if jhs_cols and not filtered_df[jhs_cols].empty else 0
             senior_high = filtered_df[shs_cols].sum().sum() if shs_cols and not filtered_df[shs_cols].empty else 0

             stage_data = pd.DataFrame({'Stage': ['Elementary', 'Junior High School', 'Senior High School'], 'Enrollment': [elementary, junior_high, senior_high]})
             stage_data = stage_data[stage_data['Enrollment'] > 0]
             if not stage_data.empty:
                  fig_education_stage = px.pie(
                      stage_data, names='Stage', values='Enrollment', title='Enrollment Distribution by Education Stage (Filtered)',
                      hole=0.4, template=PLOT_TEMPLATE, color_discrete_sequence=QUALITATIVE_COLOR_SEQUENCE)
                  fig_education_stage.update_traces(textposition='outside', textinfo='percent+label', pull=[0.05]*len(stage_data))
                  fig_education_stage.update_layout(title_font_size=14, legend_title_text='Stage', title_x=0.5, uniformtext_minsize=10, uniformtext_mode='hide', height=400, width=600)


        # 5. SHS Specific Strand Enrollment by Gender
        fig_shs_strand_gender_combined = create_placeholder_figure('SHS Enrollment by Strand and Gender')
        shs_cols_numeric = [col for col in numeric_enrollment_cols_filtered if col.startswith('G11 ') or col.startswith('G12 ')]
        if shs_cols_numeric and not filtered_df.empty:
             strand_gender_data = []
             for col in shs_cols_numeric:
                 match = re.match(r'(G11|G12) (.*) (Male|Female)$', col) # Ensure this regex is robust for all strand names
                 if match:
                      grade, strand_name, gender = match.groups()
                      if strand_name.strip() and col in filtered_df.columns: # Check col exists
                          enrollment_sum = filtered_df[col].sum()
                          if enrollment_sum > 0:
                               strand_gender_data.append({'Grade': grade, 'Strand': strand_name.strip(), 'Gender': gender, 'Enrollment': enrollment_sum})
             if strand_gender_data:
                 strand_gender_df = pd.DataFrame(strand_gender_data)
                 strand_gender_agg = strand_gender_df.groupby(['Strand', 'Gender'], observed=False)['Enrollment'].sum().reset_index()
                 if not strand_gender_agg.empty:
                     fig_shs_strand_gender_combined = px.bar(
                         strand_gender_agg, x='Strand', y='Enrollment', color='Gender', barmode='group', title='SHS Enrollment by Specific Strand/Track and Gender (Filtered)',
                         labels={'Enrollment': 'Students', 'Strand': 'Strand/Track', 'Gender': 'Gender'}, template=PLOT_TEMPLATE, color_discrete_map=GENDER_COLORS)
                     fig_shs_strand_gender_combined.update_layout(title_font_size=14, legend_title_text='Gender', title_x=0.5, height=400, width=600)
                     fig_shs_strand_gender_combined.update_xaxes(tickangle=30, categoryorder='total descending')


        # 6. Top 10 Most Populated Schools
        fig_top_schools = create_placeholder_figure('Top 10 Most Populated Schools')
        if 'BEIS School ID' in filtered_df.columns and 'School Name' in filtered_df.columns and not filtered_df.empty and numeric_enrollment_cols_filtered:
             enrollment_per_school = filtered_df.groupby(['BEIS School ID', 'School Name'])[numeric_enrollment_cols_filtered].sum().sum(axis=1).reset_index(name='Total Enrollment')
             top_schools = enrollment_per_school[enrollment_per_school['Total Enrollment'] > 0].nlargest(10, 'Total Enrollment')
             if not top_schools.empty:
                  top_schools['Display Name'] = top_schools.apply(lambda row: f"{row['School Name'][:40]}{'...' if len(row['School Name']) > 40 else ''} ({row['BEIS School ID']})", axis=1)
                  fig_top_schools = px.bar(
                      top_schools, y='Display Name', x='Total Enrollment', orientation='h', title='Top 10 Most Populated Schools (Filtered)', text='Total Enrollment',
                      labels={'Total Enrollment': 'Total Enrollment', 'Display Name': 'School'}, template=PLOT_TEMPLATE, color='Total Enrollment', color_continuous_scale=px.colors.sequential.Blues_r)
                  fig_top_schools.update_layout(title_font_size=14, yaxis={'categoryorder':'total ascending'}, title_x=0.5, height=400, width=600)
                  fig_top_schools.update_traces(textposition='outside')
                  fig_top_schools.update_coloraxes(showscale=False)


        # 7. Strand Enrollment Trend (G11 vs G12)
        fig_strand_trend = create_placeholder_figure('SHS Strand Enrollment: G11 vs G12')
        # This plot used filtered_df_base_after_geo_sector_id if grade not G11/G12
        # Or numeric_enrollment_cols_filtered and filtered_df otherwise.
        # Let's use 'all_numeric_enrollment_cols_in_base' and 'filtered_df_base_after_geo_sector_id' as the base for this trend
        # to show overall school year trend unless a specific SHS grade is selected.
        strand_trend_cols_src = all_numeric_enrollment_cols_in_base
        strand_trend_df_src = filtered_df_base_after_geo_sector_id.copy() # Use a copy

        if selected_grade in ['G11', 'G12']: # If specific SHS grade selected, use the fully filtered data
            strand_trend_cols_src = numeric_enrollment_cols_filtered # these are already filtered by G11/G12
            strand_trend_df_src = filtered_df.copy()


        g11_cols_trend = [col for col in strand_trend_cols_src if col.startswith('G11 ')]
        g12_cols_trend = [col for col in strand_trend_cols_src if col.startswith('G12 ')]

        if (g11_cols_trend or g12_cols_trend) and not strand_trend_df_src.empty:
            strand_trend_data_list = []
            strands_set = set()
            shs_pattern_trend = re.compile(r'(G11|G12)\s+(.+)\s+(Male|Female)$')

            for col in g11_cols_trend + g12_cols_trend:
                match = shs_pattern_trend.match(col)
                if match: strands_set.add(match.group(2).strip())

            for strand_name_trend in sorted(list(strands_set)):
                g11_strand_cols_trend = [col for col in g11_cols_trend if shs_pattern_trend.match(col) and shs_pattern_trend.match(col).group(2).strip() == strand_name_trend and col in strand_trend_df_src.columns]
                g12_strand_cols_trend = [col for col in g12_cols_trend if shs_pattern_trend.match(col) and shs_pattern_trend.match(col).group(2).strip() == strand_name_trend and col in strand_trend_df_src.columns]

                g11_total_trend = strand_trend_df_src[g11_strand_cols_trend].sum().sum() if g11_strand_cols_trend else 0
                g12_total_trend = strand_trend_df_src[g12_strand_cols_trend].sum().sum() if g12_strand_cols_trend else 0

                if g11_total_trend > 0: strand_trend_data_list.append({'Strand': strand_name_trend, 'Grade': 'G11', 'Enrollment': g11_total_trend})
                if g12_total_trend > 0: strand_trend_data_list.append({'Strand': strand_name_trend, 'Grade': 'G12', 'Enrollment': g12_total_trend})

            if strand_trend_data_list:
                trend_df_viz = pd.DataFrame(strand_trend_data_list)
                if not trend_df_viz.empty:
                    fig_strand_trend = px.line(
                        trend_df_viz, x='Strand', y='Enrollment', color='Grade', markers=True, title='SHS Strand Enrollment: G11 vs G12',
                        labels={'Enrollment': 'Total Students', 'Strand': 'Strand/Track', 'Grade': 'Grade Level'}, template=PLOT_TEMPLATE, color_discrete_map={'G11': '#1f77b4', 'G12': '#ff7f0e'})
                    fig_strand_trend.update_layout(title_font_size=14, legend_title_text='Grade Level', title_x=0.5, height=400, width=600)
                    fig_strand_trend.update_xaxes(tickangle=30, categoryorder='category ascending')


        # 8. Number of Schools Offering Each Strand (Treemap) - Uses filtered_df_base_after_geo_sector_id
        fig_schools_offering_strand = create_placeholder_figure('Number of Schools Offering Each SHS Strand')
        # This should use the base data for the selected school year, possibly filtered by region/division/sector but NOT by grade.
        # So filtered_df_base_after_geo_sector_id is appropriate.
        # And all_numeric_enrollment_cols_in_base for SHS cols.
        shs_cols_base_treemap = [col for col in all_numeric_enrollment_cols_in_base if col.startswith('G11 ') or col.startswith('G12 ')]

        if shs_cols_base_treemap and 'BEIS School ID' in filtered_df_base_after_geo_sector_id.columns and not filtered_df_base_after_geo_sector_id.empty:
             schools_offering_dict = {}
             df_for_treemap = filtered_df_base_after_geo_sector_id # Use the geo/sector filtered data
             for index, school_row in df_for_treemap.iterrows():
                 school_id_val = school_row['BEIS School ID']
                 if school_id_val == 'Unknown': continue
                 for col_name_treemap in shs_cols_base_treemap:
                      if col_name_treemap in school_row and pd.notna(school_row[col_name_treemap]) and school_row[col_name_treemap] > 0:
                          match_treemap = re.match(r'(G11|G12) (.*) (Male|Female)$', col_name_treemap) # Original regex
                          if match_treemap:
                               # Original code was " ".join(match.groups()[1:-1]) which might be problematic if strand name has spaces
                               # Assuming strand name is group 2 as per most regexes.
                               strand_name_treemap = match_treemap.group(2).strip()
                               if strand_name_treemap:
                                    if strand_name_treemap not in schools_offering_dict:
                                        schools_offering_dict[strand_name_treemap] = set()
                                    schools_offering_dict[strand_name_treemap].add(school_id_val)
             if schools_offering_dict:
                  strand_counts_list = [{'Strand': s_name, 'Schools Offering': len(s_ids)} for s_name, s_ids in schools_offering_dict.items()]
                  strand_counts_df_viz = pd.DataFrame(strand_counts_list).sort_values('Schools Offering', ascending=False)
                  if not strand_counts_df_viz.empty:
                       fig_schools_offering_strand = px.treemap(
                           strand_counts_df_viz, path=[px.Constant("All Strands"), 'Strand'], values='Schools Offering', title='Number of Schools Offering Each SHS Strand/Track',
                           labels={'Schools Offering': 'Count of Schools', 'Strand': 'Strand/Track'}, template=PLOT_TEMPLATE, color='Schools Offering', color_continuous_scale=px.colors.sequential.YlGnBu)
                       fig_schools_offering_strand.update_layout(title_font_size=14, title_x=0.5, height=400, width=600)
                       fig_schools_offering_strand.update_traces(textinfo="label+value")

        # Watchlist Table
        watchlist_table_component = html.P("No schools data for watchlist based on filters.", style={"textAlign": "center"})
        if 'BEIS School ID' in filtered_df.columns and 'School Name' in filtered_df.columns and not filtered_df.empty and numeric_enrollment_cols_filtered:
             # Uses numeric_enrollment_cols_filtered, which is grade-specific if a grade is selected
             enrollment_per_school_watchlist = filtered_df.groupby(['BEIS School ID', 'School Name'], observed=False)[numeric_enrollment_cols_filtered].sum().sum(axis=1).reset_index(name='Filtered Total Enrollment')
             least_populated_df = enrollment_per_school_watchlist[enrollment_per_school_watchlist['Filtered Total Enrollment'] > 0].nsmallest(10, 'Filtered Total Enrollment')

             if not least_populated_df.empty:
                  cols_to_merge_watchlist = ['BEIS School ID', 'Region', 'Division', 'Sector', 'School Type', 'School Subclassification']
                  # Merge from the broader filtered_df (before grade specific numeric cols were applied for sum)
                  # or from df_all if that's more appropriate for context.
                  # Original used filtered_df, which is already geo/sector/id filtered.
                  merge_info_df_watchlist = filtered_df[[col for col in cols_to_merge_watchlist if col in filtered_df.columns]].drop_duplicates(subset=['BEIS School ID'])
                  least_populated_df = pd.merge(least_populated_df, merge_info_df_watchlist, on='BEIS School ID', how='left')

                  table_cols_order_watchlist = ['BEIS School ID', 'School Name', 'Filtered Total Enrollment', 'Sector', 'Division', 'Region', 'School Type', 'School Subclassification']
                  final_table_cols_watchlist = [col for col in table_cols_order_watchlist if col in least_populated_df.columns]
                  least_populated_df = least_populated_df[final_table_cols_watchlist]

                  watchlist_table_component = dash_table.DataTable(
                      data=least_populated_df.to_dict("records"),
                      columns=[{"name": i.replace('Filtered Total Enrollment', 'Total Enrollment (Filtered)'), "id": i} for i in least_populated_df.columns],
                      page_size=10, style_table={'overflowX': 'auto'},
                      style_cell={'textAlign': 'left', 'fontSize': '0.8rem', 'padding': '5px', 'fontFamily': 'Arial, sans-serif', 'whiteSpace': 'normal', 'height': 'auto'},
                      style_header={'fontWeight': 'bold', 'backgroundColor': '#f0f0f0'},
                      style_data_conditional=[{'if': {'row_index': 'odd'}, 'backgroundColor': '#f9f9f9'}],
                      sort_action="native", filter_action="native", id='watchlist-table'
                  )

        return (kpis, fig_region_bar, fig_grade_gender_parity, fig_sector, fig_education_stage,
                fig_shs_strand_gender_combined, fig_top_schools, fig_strand_trend,
                fig_schools_offering_strand, watchlist_table_component)


    @dash_app_report.callback(
        Output("download-data", "data"),
        Input("btn-download", "n_clicks"),
        State('main-data-store', 'data'), # Get the base data for the school year
        State('enrollment-cols-store', 'data'), # Get all enrollment cols for this dataset
        State('region-filter', 'value'),
        State('division-filter', 'value'),
        State('grade-filter', 'value'), # This is key for filtering columns
        State('sector-filter', 'value'),
        State('beis-id-filter', 'value'),
        prevent_initial_call=True,
    )
    def download_filtered_data(n_clicks, df_json, all_enrollment_cols_for_sy, # Renamed for clarity
                               selected_region, selected_division, selected_grade,
                               selected_sector, selected_beis_id):
        if not df_json or not all_enrollment_cols_for_sy:
            return None # Or dict(content="No data to download.", filename="empty_data.txt")

        df_all_sy = pd.read_json(df_json, orient='split') # Data for the selected school year

        # Apply filters to rows
        filtered_df_download = df_all_sy.copy()
        query_parts_dl = []
        if selected_region and selected_region != 'Unknown': query_parts_dl.append(f"Region == '{selected_region}'")
        if selected_division and selected_division != 'Unknown': query_parts_dl.append(f"Division == '{selected_division}'")
        if selected_sector and selected_sector != 'Unknown': query_parts_dl.append(f"Sector == '{selected_sector}'")
        if selected_beis_id and selected_beis_id != 'Unknown': query_parts_dl.append(f"`BEIS School ID` == '{str(selected_beis_id)}'")

        if query_parts_dl:
            try:
                filtered_df_download = filtered_df_download.query(" and ".join(query_parts_dl))
            except Exception as e:
                print(f"Error querying for download: {e}")
                # Decide behavior: download unfiltered for school year, or error out
                # For now, proceed with what's filtered so far.

        # Identify non-enrollment columns from the loaded school year dataframe
        non_enrollment_cols_dl = [col for col in df_all_sy.columns if col not in all_enrollment_cols_for_sy]

        # Filter columns by selected_grade
        columns_to_download_final = non_enrollment_cols_dl # Start with non-enrollment columns
        if selected_grade:
            grade_specific_enrollment_cols = []
            for col_enrl in all_enrollment_cols_for_sy: # Use the full list for the SY
                 if col_enrl in filtered_df_download.columns: # Ensure column exists in (potentially row-filtered) df
                     if col_enrl.startswith(selected_grade + " ") or \
                        (selected_grade == 'K' and col_enrl.startswith('K ')):
                         grade_specific_enrollment_cols.append(col_enrl)
            columns_to_download_final.extend(grade_specific_enrollment_cols)
        else: # No specific grade selected, include all enrollment columns for the school year
            columns_to_download_final.extend(all_enrollment_cols_for_sy)

        columns_to_download_final = [col for col in list(dict.fromkeys(columns_to_download_final)) if col in filtered_df_download.columns]
        
        df_to_output = filtered_df_download[columns_to_download_final]

        csv_buffer = io.StringIO()
        df_to_output.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_buffer.seek(0)
        
        return dict(content=csv_buffer.getvalue(), filename=f"filtered_enrollment_data_{selected_grade if selected_grade else 'all_grades'}.csv")

    return dash_app_report