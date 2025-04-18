from dash import Dash, html, dcc, Input, Output, State
import plotly.express as px
import pandas as pd
from dash import dash_table
from data_config import get_dataset_path, fetch_enrollment_records_from_csv, fetch_summary_data_from_csv
import io
import base64

def create_dash_app_report(flask_app):
    dash_app_report = Dash(__name__, server=flask_app, routes_pathname_prefix="/dashreport/", external_stylesheets=['assets/style.css'])

    file_path = get_dataset_path()
    all_data = fetch_enrollment_records_from_csv(file_path)
    df_all = pd.DataFrame(all_data)
    summary_all = fetch_summary_data_from_csv(file_path)

    # Extract unique values for filters
    regions = sorted(df_all["Region"].unique()) if "Region" in df_all.columns else []
    divisions_by_region = df_all.groupby("Region")["Division"].unique().apply(sorted).to_dict() if "Region" in df_all.columns and "Division" in df_all.columns else {}
    all_divisions = sorted(df_all["Division"].unique()) if "Division" in df_all.columns else []
    beis_ids_by_region_df = df_all.groupby("Region")[["BEIS School ID", "School Name"]].apply(lambda x: sorted(x.set_index("BEIS School ID")["School Name"].to_dict().items())).to_dict() if "Region" in df_all.columns and "BEIS School ID" in df_all.columns and "School Name" in df_all.columns else {}
    all_beis_ids_with_names = sorted(df_all[["BEIS School ID", "School Name"]].set_index("BEIS School ID")["School Name"].to_dict().items()) if "BEIS School ID" in df_all.columns and "School Name" in df_all.columns else []
    all_beis_ids = [{'label': f"{id_} - {name}", 'value': id_} for id_, name in all_beis_ids_with_names]
    grades_all_temp = sorted(list(set([col.replace(" Male", "").replace(" Female", "").strip() for col in df_all.columns if any(g in col for g in ['K', 'G1', 'G2', 'G3', 'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12'])])))
    ordered_grades = ['K'] + [f'G{i}' for i in range(1, 11)] + ['G11', 'G12']
    grades = ordered_grades
    sector_types = sorted(df_all["Sector"].unique()) if "Sector" in df_all.columns else []

    dash_app_report.layout = html.Div([
        html.H1("📊Looking for enrollment data? Find what you need right here.", style={"textAlign": "center", "marginBottom": "20px", "color": "#333", "fontSize": "2rem"}),

        # Filters Section
        html.Div([
            html.Div([
                html.Label("🔍 Region", className="filter-label", style={"fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id='region-filter',
                    options=[{'label': r, 'value': r} for r in regions],
                    value=None,
                    placeholder="Select Region",
                    className="dropdown",
                    style={"fontSize": "0.8rem"}
                ),
            ], className="filter-item"),

            html.Div([
                html.Label("📍 Division", className="filter-label", style={"fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id='division-filter',
                    options=[{'label': d, 'value': d} for d in all_divisions],
                    value=None,
                    placeholder="Select Division",
                    className="dropdown",
                    style={"fontSize": "0.8rem"}
                ),
            ], className="filter-item"),

            html.Div([
                html.Label("🎓 Grade Level", className="filter-label", style={"fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id='grade-filter',
                    options=[{'label': g, 'value': g} for g in grades],
                    value=None,
                    placeholder="Select Grade Level",
                    className="dropdown",
                    style={"fontSize": "0.8rem"}
                ),
            ], className="filter-item"),

            html.Div([
                html.Label("🏫 Sector Type", className="filter-label", style={"fontSize": "0.85rem"}),
                dcc.RadioItems(
                    id='sector-filter',
                    options=[{'label': s, 'value': s} for s in sector_types],
                    value=None,
                    inline=True,
                    className="radio-items",
                    style={"fontSize": "0.8rem"}
                ),
            ], className="filter-item"),

            html.Div([
                html.Label("🔑 BEIS School ID", className="filter-label", style={"fontSize": "0.85rem"}),
                dcc.Dropdown(
                    id='beis-id-filter',
                    options=all_beis_ids,
                    value=None,
                    placeholder="Select BEIS School ID",
                    className="dropdown",
                    style={"fontSize": "0.8rem"}
                ),
            ], className="filter-item"),
        ], className="filters-container", style={"padding": "20px", "gap": "15px"}),

        html.Div(style={"display": "flex", "justifyContent": "center", "marginBottom": "15px"}),
        html.Button("Reset Filters", id="reset-button", n_clicks=0, className="reset-button", style={"backgroundColor": "#4CAF50", "color": "white", "padding": "8px 15px", "fontSize": "0.8rem", "marginRight": "10px"}),
        html.Button("⬇ Download Filtered Data", id="btn-download", n_clicks=0, className="download-button", style={"backgroundColor": "#008CBA", "color": "white", "padding": "8px 15px", "fontSize": "0.8rem", "marginLeft": "10px"}),
        html.Div(style={"display": "flex", "justifyContent": "center"}),

        html.Hr(style={"marginTop": "15px", "marginBottom": "25px", "borderColor": "#ddd"}),

        # KPI Cards Section with Loading
        dcc.Loading(
            id="loading-kpi",
            type="circle",
            children=html.Div(id='kpi-cards', className="kpi-cards-container")
        ),

        html.Hr(style={"marginTop": "25px", "marginBottom": "25px", "borderColor": "#ddd"}),

        # Main Visualizations Section with Loading
        dcc.Loading(
            id="loading-graphs",
            type="circle",
            children=html.Div([
                dcc.Graph(id='region-enrollment-bar', className="graph-item"),
                dcc.Graph(id='grade-gender-parity-bar', className="graph-item")
            ], className="row", style={"gap": "20px"})
        ),

        dcc.Loading(
            id="loading-graphs-2",
            type="circle",
            children=html.Div([
                dcc.Graph(id='sector-distribution', className="graph-item"),
                dcc.Graph(id='education-stage-distribution', className="graph-item")
            ], className="row", style={"gap": "20px"})
        ),

        html.H2("⚠️ Watchlist: Schools Under This", style={"marginTop": "30px", "marginBottom": "12px", "color": "#d32f2f", "fontSize": "1.4rem"}),
        dcc.Loading(
            id="loading-table",
            type="circle",
            children=html.Div(id='flagged-schools-table', className="table-container", style={"padding": "15px", "fontSize": "0.8rem"})
        ),

        html.Br(),

        dcc.Download(id="download-data")
    ], className="main-container", style={"backgroundColor": "#f9f9f9", "padding": "40px"})

    # Callback to update Division based on selected Region
    @dash_app_report.callback(
        Output('division-filter', 'options'),
        Input('region-filter', 'value')
    )
    def update_divisions(selected_region):
        if selected_region:
            return [{'label': d, 'value': d} for d in divisions_by_region.get(selected_region, [])]
        return [{'label': d, 'value': d} for d in all_divisions]

    # Callback to update BEIS School ID based on selected Region
    @dash_app_report.callback(
        Output('beis-id-filter', 'options'),
        Input('region-filter', 'value')
    )
    def update_beis_ids(selected_region):
        if selected_region:
            beis_ids_with_names_in_region = beis_ids_by_region_df.get(selected_region, [])
            return [{'label': f"{id_} - {name}", 'value': id_} for id_, name in beis_ids_with_names_in_region]
        return all_beis_ids

    # Callback for resetting all filters
    @dash_app_report.callback(
        Output('region-filter', 'value'),
        Output('division-filter', 'value'),
        Output('grade-filter', 'value'),
        Output('sector-filter', 'value'),
        Output('beis-id-filter', 'value'),
        Input('reset-button', 'n_clicks'),
        State('region-filter', 'value'),
        State('division-filter', 'value'),
        State('grade-filter', 'value'),
        State('sector-filter', 'value'),
        State('beis-id-filter', 'value'),
    )
    def reset_filters(n_clicks, region_val, division_val, grade_val, sector_val, beis_id_val):
        if n_clicks > 0:
            return None, None, None, None, None
        return region_val, division_val, grade_val, sector_val, beis_id_val

    @dash_app_report.callback(
        Output('kpi-cards', 'children'),
        Output('region-enrollment-bar', 'figure'),
        Output('grade-gender-parity-bar', 'figure'),
        Output('sector-distribution', 'figure'),
        Output('education-stage-distribution', 'figure'), # New output
        Output('flagged-schools-table', 'children'),
        Input('region-filter', 'value'),
        Input('division-filter', 'value'),
        Input('grade-filter', 'value'),
        Input('sector-filter', 'value'),
        Input('beis-id-filter', 'value'),
    )
    def update_dashboard(selected_region, selected_division, selected_grade, selected_sector, selected_beis_id):
        filtered_df_base = df_all.copy()

        if selected_region:
            filtered_df_base = filtered_df_base[filtered_df_base['Region'] == selected_region]
        if selected_division:
            filtered_df_base = filtered_df_base[filtered_df_base['Division'] == selected_division]
        if selected_sector:
            filtered_df_base = filtered_df_base[filtered_df_base['Sector'] == selected_sector]
        if selected_beis_id:
            filtered_df_base = filtered_df_base[filtered_df_base['BEIS School ID'] == selected_beis_id]

        # Most Populated Year Level Calculation (using the base filtered DataFrame)
        grade_enrollment = {}
        grade_cols_all = [col for col in filtered_df_base.columns if any(g in col for g in ['K'] + [f'G{i}' for i in range(1, 13)]) and ("Male" in col or "Female" in col)]
        for col in grade_cols_all:
            grade = col.replace(" Male", "").replace(" Female", "").strip()
            grade_match = None
            if grade.startswith('K'):
                grade_match = 'K'
            elif grade.startswith('G') and grade[1:].isdigit():
                grade_match = grade
            if grade_match:
                if grade_match not in grade_enrollment:
                    grade_enrollment[grade_match] = 0
                grade_enrollment[grade_match] += filtered_df_base[col].sum()

        most_populated_grade = "" # Initialize with an empty string
        if grade_enrollment:
            most_populated_grade = max(grade_enrollment, key=grade_enrollment.get)

        filtered_df = filtered_df_base.copy()

        # Grade Filtering (Includes check for G11 and G12 with strand info)
        grade_columns_to_keep = []
        if selected_grade:
            for col in filtered_df.columns:
                if selected_grade in ['G11', 'G12']:
                    if selected_grade in col and ("Male" in col or "Female" in col):
                        grade_columns_to_keep.append(col)
                elif col.startswith(f"{selected_grade} Male") or col.startswith(f"{selected_grade} Female"):
                    grade_columns_to_keep.append(col)
            filtered_df = filtered_df[[col for col in filtered_df.columns if col in ['Region', 'Division', 'District', 'BEIS School ID', 'School Name', 'Street Address', 'Province', 'Municipality', 'Legislative District', 'Barangay', 'Sector', 'School Subclassification', 'School Type', 'Modified COC']] + grade_columns_to_keep]

        # Recalculate summary based on filtered data
        if selected_grade:
            grade_cols_for_total = [col for col in filtered_df.columns if selected_grade in col and ("Male" in col or "Female" in col)]
            total_enrollments = filtered_df[grade_cols_for_total].sum().sum() if grade_cols_for_total else 0
            male_enrollments = filtered_df[[col for col in grade_cols_for_total if 'Male' in col]].sum().sum() if any('Male' in col for col in grade_cols_for_total) else 0
            female_enrollments = filtered_df[[col for col in grade_cols_for_total if 'Female' in col]].sum().sum() if any('Female' in col for col in grade_cols_for_total) else 0
        else:
            total_enrollments = filtered_df[[col for col in filtered_df.columns if 'K' in col or 'G' in col]].sum().sum()
            male_enrollments = filtered_df[[col for col in filtered_df.columns if 'Male' in col and ('K' in col or 'G' in col)]].sum().sum()
            female_enrollments = filtered_df[[col for col in filtered_df.columns if 'Female' in col and ('K' in col or 'G' in col)]].sum().sum()
        number_of_schools = filtered_df['BEIS School ID'].nunique() if 'BEIS School ID' in filtered_df.columns else 0

        summary_filtered = {
            'totalEnrollments': total_enrollments,
            'maleEnrollments': male_enrollments,
            'femaleEnrollments': female_enrollments,
            'numberOfSchools': number_of_schools
        }

        kpis = html.Div([
            html.Div([
                html.H3("Total Enrolled Learners", className="kpi-title", style={"fontSize": "1rem"}),
                html.H1(f"{summary_filtered.get('totalEnrollments', 0):,}", className="kpi-value", style={"fontSize": "1.6rem"})
            ], className="kpi-card", style={"padding": "20px"}),
            html.Div([
                html.H3("Male vs Female", className="kpi-title", style={"fontSize": "1rem"}),
                html.P(f"{summary_filtered.get('maleEnrollments', 0):,} ♂ | {summary_filtered.get('femaleEnrollments', 0):,} ♀", className="kpi-value", style={"fontSize": "1.6rem", "textAlign": "center"})
            ], className="kpi-card", style={"padding": "20px"}),
            html.Div([
                html.H3("No. of Schools", className="kpi-title", style={"fontSize": "1rem"}),
                html.H1(f"{summary_filtered.get('numberOfSchools', 0):,}", className="kpi-value", style={"fontSize": "1.6rem"})
            ], className="kpi-card", style={"padding": "20px"}),
            html.Div([
                html.H3("Most Populated Year Level", className="kpi-title", style={"fontSize": "1rem"}),
                html.H1(most_populated_grade, className="kpi-value", style={"fontSize": "1.6rem"})
            ], className="kpi-card", style={"padding": "20px"}),
        ], className="kpi-cards-container", style={"gap": "20px", "padding": "10px 0"})

        # Bar chart for male/female parity per grade level
        grade_columns = []
        if selected_grade:
            # Use 'in' to find the selected grade in the column name
            for col in filtered_df.columns:
                if selected_grade in col and ("Male" in col or "Female" in col):
                    grade_columns.append(col)
        else:
            grade_columns = [col for col in filtered_df.columns if any(g == col.split(' ')[0] for g in ['K'] + [f'G{i}' for i in range(1, 13)]) and ("Male" in col or "Female" in col)]

        if grade_columns:
            melted_filtered = pd.melt(filtered_df[grade_columns], var_name="GradeGender", value_name="Count")
            # Use a regex that includes G11 and G12
            melted_filtered["Grade"] = melted_filtered["GradeGender"].str.extract(r'(K|G\d{1,2}|G11|G12)').ffill()
            melted_filtered["Gender"] = melted_filtered["GradeGender"].str.extract(r'(Male|Female)$')

            parity_fig = px.bar(
                melted_filtered.groupby(["Grade", "Gender"])["Count"].sum().reset_index(),
                x="Grade", y="Count", color="Gender", barmode="group",
                title="Male vs Female Enrollment by Grade Level"
            )
            parity_fig.update_xaxes(categoryorder='array', categoryarray=['K'] + [f'G{i}' for i in range(1, 13)])
            parity_fig.update_layout(title_font_size=14)
        else:
            parity_fig = px.bar(title="Male vs Female Enrollment by Grade Level (No Data)")
            parity_fig.update_layout(title_font_size=14)

        # Bar graph for enrollment per region
        region_enrollment = filtered_df.groupby("Region")[[col for col in filtered_df.columns if 'K' in col or 'G' in col]].sum().sum(axis=1).reset_index(name='Total Enrollment')
        fig_region_bar = px.bar(region_enrollment, x="Region", y="Total Enrollment", title="Enrollment per Region")
        fig_region_bar.update_layout(title_font_size=14)

        # Sector Type Distribution
        if 'Sector' in filtered_df.columns:
            sector_counts = filtered_df['Sector'].value_counts().reset_index()
            sector_counts.columns = ['Sector', 'Count']
            fig_sector = px.pie(sector_counts, names='Sector', values='Count', title='Enrollment by Sector Type')
            fig_sector.update_layout(title_font_size=14)
        else:
            fig_sector = px.pie(title='Enrollment by Sector Type (No Data)')
            fig_sector.update_layout(title_font_size=14)

        # Education Stage Distribution
        elementary_grades = ['K'] + [f'G{i}' for i in range(1, 7)] # Assuming Elem NG is included in these grades
        junior_high_grades = [f'G{i}' for i in range(7, 11)] # Assuming JHS NG is included in these grades
        senior_high_grades = ['G11', 'G12']

        elementary_enrollment = filtered_df[[col for col in filtered_df.columns if any(grade in col.split(' ')[0] for grade in elementary_grades) and ("Male" in col or "Female" in col)]].sum().sum()
        junior_high_enrollment = filtered_df[[col for col in filtered_df.columns if any(grade in col.split(' ')[0] for grade in junior_high_grades) and ("Male" in col or "Female" in col)]].sum().sum()
        senior_high_enrollment = filtered_df[[col for col in filtered_df.columns if any(grade in col.split(' ')[0] for grade in senior_high_grades) and ("Male" in col or "Female" in col)]].sum().sum()

        education_stage_data = pd.DataFrame({
            'Stage': ['Elementary', 'Junior High School', 'Senior High School'],
            'Enrollment': [elementary_enrollment, junior_high_enrollment, senior_high_enrollment]
        })

        fig_education_stage = px.pie(education_stage_data, names='Stage', values='Enrollment', title='Enrollment by Education Stage')
        fig_education_stage.update_layout(title_font_size=14)

        flagged_schools_filtered = filtered_df[filtered_df["K Male"] < 10] if "K Male" in filtered_df.columns else pd.DataFrame()
        flagged_schools_table = dash_table.DataTable(
            data=flagged_schools_filtered.to_dict("records"),
            columns=[{"name": i, "id": i} for i in flagged_schools_filtered.columns], # Added columns definition
            page_size=10,
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left', 'fontSize': '0.8rem'}
        )

        return kpis, fig_region_bar, parity_fig, fig_sector, fig_education_stage, flagged_schools_table

    @dash_app_report.callback(
        Output("download-data", "data"),
        Input("btn-download", "n_clicks"),
        State('region-filter', 'value'),
        State('division-filter', 'value'),
        State('grade-filter', 'value'),
        State('sector-filter', 'value'),
        State('beis-id-filter', 'value'),
        prevent_initial_call=True,
    )
    def download_filtered_data(n_clicks, selected_region, selected_division, selected_grade, selected_sector, selected_beis_id):
        filtered_df = df_all.copy()

        if selected_region:
            filtered_df = filtered_df[filtered_df['Region'] == selected_region]
        if selected_division:
            filtered_df = filtered_df[filtered_df['Division'] == selected_division]
        if selected_sector:
            filtered_df = filtered_df[filtered_df['Sector'] == selected_sector]
        if selected_beis_id:
            filtered_df = filtered_df[filtered_df['BEIS School ID'] == selected_beis_id]

        grade_columns_to_keep = []
        if selected_grade:
            for col in filtered_df.columns:
                if selected_grade in ['G11', 'G12']:
                    if selected_grade in col and ("Male" in col or "Female" in col):
                        grade_columns_to_keep.append(col)
                elif col.startswith(f"{selected_grade} Male") or col.startswith(f"{selected_grade} Female"):
                    grade_columns_to_keep.append(col)
            if grade_columns_to_keep:
                columns_to_download = ['Region', 'Division', 'District', 'BEIS School ID', 'School Name', 'Street Address', 'Province', 'Municipality', 'Legislative District', 'Barangay', 'Sector', 'School Subclassification', 'School Type', 'Modified COC'] + grade_columns_to_keep
                filtered_df = filtered_df[columns_to_download]
            else:
                filtered_df = filtered_df[['Region', 'Division', 'District', 'BEIS School ID', 'School Name', 'Street Address', 'Province', 'Municipality', 'Legislative District', 'Barangay', 'Sector', 'School Subclassification', 'School Type', 'Modified COC']]
        else:
            filtered_df = filtered_df

        csv_buffer = io.StringIO()
        filtered_df.to_csv(csv_buffer, index=False, encoding='utf-8')
        csv_buffer.seek(0)
        csv_data = csv_buffer.getvalue()

        return dict(content=csv_data, filename="filtered_enrollment_data.csv")
    return dash_app_report