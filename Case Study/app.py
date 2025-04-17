from flask import Flask, render_template, request, redirect, flash, jsonify, url_for, send_file
import os
from works import create_dash_app
from werkzeug.utils import secure_filename
from data_config import get_dataset_path, fetch_enrollment_records_from_csv, fetch_summary_data_from_csv
from data_cleaning import clean_data
from datetime import datetime
from report import create_dash_app_report
import shutil

app = Flask(__name__)
app.secret_key = 'secret123'

# Upload folder config
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static')
CLEANED_FOLDER = os.path.join(os.path.dirname(__file__), 'cleaned_files')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CLEANED_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("account.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route("/otheryear")
def otheryear():
    return render_template("otheryear.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

@app.route("/report")
def report():
    return render_template("report.html")

@app.route("/help")
def help():
    return render_template("help.html")

@app.route("/logout")
def logout():
    return render_template("logout.html")

@app.route("/upload", methods=["GET", "POST"])
def upload():
    dataset_path = os.path.join(UPLOAD_FOLDER, 'Cleaned_School_DataSet.csv')
    last_updated = None
    if os.path.exists(dataset_path):
        last_updated = datetime.fromtimestamp(os.path.getmtime(dataset_path)).strftime('%Y-%m-%d %H:%M:%S')

    if request.method == "POST":
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            try:
                # Save raw uploaded file temporarily
                raw_filename = secure_filename(file.filename)
                raw_path = os.path.join(CLEANED_FOLDER, f"temp_{raw_filename}")
                file.save(raw_path)

                # Clean the file
                cleaned_path = clean_data(raw_path)

                shutil.copy(cleaned_path, dataset_path)

                flash('CSV file uploaded, cleaned, and is now active!')
                last_updated = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            except Exception as e:
                flash(f"Data cleaning failed: {str(e)}")
                return redirect(request.url)

            return render_template("upload.html", last_updated=last_updated)

        flash("Invalid file type. Please upload a .csv file.")
        return redirect(request.url)

    return render_template("upload.html", last_updated=last_updated)

# Clean raw dataset and auto-download
@app.route('/clean', methods=['GET', 'POST'])
def clean():
    if request.method == 'POST':
        uploaded_file = request.files.get('uncleaned_file')
        if uploaded_file and uploaded_file.filename != '':
            filename = secure_filename(uploaded_file.filename)
            raw_path = os.path.join(CLEANED_FOLDER, filename)
            uploaded_file.save(raw_path)

            try:
                cleaned_path = clean_data(raw_path)
                return send_file(cleaned_path, as_attachment=True)

            except Exception as e:
                flash(f"Data cleaning failed: {str(e)}")
                return redirect(request.url)

        flash("No valid file selected for cleaning.")
        return redirect(request.url)

    return render_template('upload.html')

@app.route('/api/enrollment_data')
@app.route('/api/enrollment_data')
def get_enrollment_data():
    file_path = get_dataset_path()
    data = fetch_enrollment_records_from_csv(file_path) 
    data = fetch_summary_data_from_csv(file_path)
    return jsonify(data)

# Mount Dash app
dash_app_works = create_dash_app(app)
dash_app_report = create_dash_app_report(app)

if __name__ == "__main__":
    app.run(debug=True)