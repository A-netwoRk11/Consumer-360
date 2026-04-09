# Consumer360 - Flask Data Analytics Dashboard

Consumer360 is a Flask-based analytics dashboard for retail customer intelligence.
It combines KPI tracking, RFM segmentation, sales trends, and cohort retention into a clean multi-page web app. 

##https://consumer-360.onrender.com/

## What It Solves

- Turns raw transaction and segmentation CSV files into business-ready insights.
- Makes customer behavior visible through segment and retention views.
- Supports filtering by date, country, and segment for focused analysis.

## Final Project Structure

```
consumer360_project/
|
|-- app/
|   |-- app.py
|   |-- templates/
|   |   |-- base.html
|   |   |-- index.html
|   |   |-- rfm.html
|   |   |-- sales.html
|   |   `-- cohort.html
|   `-- static/
|       |-- sales_trend.png
|       |-- rfm_segments.png
|       `-- cohort_heatmap.png
|
|-- data/
|   |-- cleaned_transactions.csv
|   `-- rfm_segments.csv
|
|-- src/
|-- requirements.txt
|-- Procfile
`-- README.md
```

## Tech Stack

- Flask
- pandas
- matplotlib
- numpy
- seaborn
- mlxtend

## Local Run

1. Create and activate virtual environment (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the Flask app:

```bash
python app/app.py
```

4. Open in browser:

`http://127.0.0.1:5000`

## Routes

- `/` - Main dashboard
- `/segments` - RFM segmentation page
- `/sales` - Sales analytics page
- `/cohort` - Cohort analysis page

## Render Deployment

This project is ready for Render.

- Build command:

```bash
pip install -r requirements.txt
```

- Start command (from Procfile):

```bash
web: python app/app.py
```

The app uses dynamic port binding:

```python
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
```

## Notes

- Charts are generated with matplotlib and saved in `app/static/`.
- Templates use Flask `url_for('static', filename=...)` for static asset loading.

