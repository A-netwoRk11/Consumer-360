import base64
import io
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import pandas as pd
from flask import Flask, render_template, request


matplotlib.use("Agg")

# Get the correct base directory - handles both local and deployed environments
BASE_DIR = Path(__file__).resolve().parent
APP_ROOT = BASE_DIR.parent

# Try to locate data directory - check parent first, then current
if (APP_ROOT / "data").exists():
    DATA_DIR = APP_ROOT / "data"
else:
    DATA_DIR = BASE_DIR / "data"

TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

print(f"[DEBUG] BASE_DIR: {BASE_DIR}")
print(f"[DEBUG] APP_ROOT: {APP_ROOT}") 
print(f"[DEBUG] DATA_DIR: {DATA_DIR}")
print(f"[DEBUG] DATA_DIR exists: {DATA_DIR.exists()}")

TRANSACTIONS_FILE = "cleaned_transactions.csv"
RFM_FILE = "rfm_segments.csv"

PRIMARY_COLOR = "#4f46e5"
ACCENT_COLOR = "#7c3aed"
GRID_COLOR = "#d9def7"

app = Flask(
	__name__,
	template_folder=str(TEMPLATES_DIR),
	static_folder=str(STATIC_DIR),
)

# Startup validation
def validate_startup():
	"""Check if required files exist on startup"""
	required_files = [
		DATA_DIR / TRANSACTIONS_FILE,
		DATA_DIR / RFM_FILE,
	]
	
	missing = [f for f in required_files if not f.exists()]
	
	if missing:
		print(f"[WARNING] Missing required data files on startup:")
		for f in missing:
			print(f"  - {f}")
	else:
		print(f"[INFO] All required data files found")

# Run validation on startup
try:
	validate_startup()
except Exception as e:
	print(f"[ERROR] Startup validation failed: {e}")


def load_data(filename):
    try:
        path = DATA_DIR / filename
        print(f"[DEBUG] Loading file: {path} (exists: {path.exists()})")

        if not path.exists():
            print(f"[WARNING] File not found: {path}")
            return pd.DataFrame()

        data = pd.read_csv(path)
        print(f"[DEBUG] Successfully loaded {filename}: {len(data)} rows")
        return data

    except Exception as e:
        print(f"[ERROR] Error loading file {filename}: {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def clean_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
	if transactions.empty:
		return transactions

	cleaned = transactions.copy()
	cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
	return cleaned.dropna(subset=["InvoiceDate"])


def calculate_kpis(transactions: pd.DataFrame) -> dict[str, float | int]:
	if transactions.empty:
		return {
			"total_revenue": 0.0,
			"total_customers": 0,
			"average_order_value": 0.0,
		}

	total_revenue = float(transactions["TotalAmount"].sum())
	total_customers = int(transactions["CustomerID"].nunique())
	total_orders = int(transactions["InvoiceID"].nunique())
	average_order_value = total_revenue / total_orders if total_orders else 0.0

	return {
		"total_revenue": total_revenue,
		"total_customers": total_customers,
		"average_order_value": average_order_value,
	}


def parse_filters() -> dict[str, str]:
	return {
		"start_date": request.args.get("start_date", "").strip(),
		"end_date": request.args.get("end_date", "").strip(),
		"segment": request.args.get("segment", "").strip(),
		"country": request.args.get("country", "").strip(),
	}


def apply_filters(
	transactions: pd.DataFrame,
	rfm: pd.DataFrame,
	filters: dict[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
	filtered_transactions = transactions.copy()

	country = filters["country"]
	if country:
		filtered_transactions = filtered_transactions[filtered_transactions["Country"] == country]

	start_date = pd.to_datetime(filters["start_date"], errors="coerce") if filters["start_date"] else None
	end_date = pd.to_datetime(filters["end_date"], errors="coerce") if filters["end_date"] else None

	if start_date is not None and pd.notna(start_date):
		filtered_transactions = filtered_transactions[filtered_transactions["InvoiceDate"] >= start_date]
	if end_date is not None and pd.notna(end_date):
		filtered_transactions = filtered_transactions[filtered_transactions["InvoiceDate"] <= end_date]

	filtered_rfm = rfm.copy()
	if not filtered_rfm.empty and "CustomerID" in filtered_rfm.columns:
		active_customers = filtered_transactions["CustomerID"].astype(str).unique()
		filtered_rfm = filtered_rfm[filtered_rfm["CustomerID"].astype(str).isin(active_customers)]

	segment = filters["segment"]
	if segment and not filtered_rfm.empty and "Segment" in filtered_rfm.columns:
		filtered_rfm = filtered_rfm[filtered_rfm["Segment"] == segment]
		segment_customers = filtered_rfm["CustomerID"].astype(str).unique()
		filtered_transactions = filtered_transactions[
			filtered_transactions["CustomerID"].astype(str).isin(segment_customers)
		]

	return filtered_transactions, filtered_rfm


def _fig_to_b64(fig) -> str:
	"""Convert a matplotlib figure to a base64 PNG data URI."""
	buf = io.BytesIO()
	fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
	plt.close(fig)
	buf.seek(0)
	encoded = base64.b64encode(buf.read()).decode("utf-8")
	return f"data:image/png;base64,{encoded}"


def generate_sales_chart(transactions: pd.DataFrame) -> str | None:
	if transactions.empty:
		return None

	monthly = (
		transactions.assign(YearMonth=transactions["InvoiceDate"].dt.to_period("M").dt.to_timestamp())
		.groupby("YearMonth", as_index=False)["TotalAmount"]
		.sum()
		.rename(columns={"TotalAmount": "Revenue"})
		.sort_values("YearMonth")
	)

	fig, ax = plt.subplots(figsize=(13, 6))
	ax.plot(
		monthly["YearMonth"],
		monthly["Revenue"],
		marker="o",
		markersize=6,
		linewidth=2.6,
		color=PRIMARY_COLOR,
		markerfacecolor=ACCENT_COLOR,
	)
	ax.set_title("Monthly Revenue Trend", fontsize=14, fontweight="bold")
	ax.set_xlabel("Month", fontsize=11)
	ax.set_ylabel("Revenue (USD)", fontsize=11)
	ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
	ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
	ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
	ax.grid(True, axis="both", linestyle="--", linewidth=0.8, color=GRID_COLOR, alpha=0.9)
	ax.tick_params(axis="x", rotation=35)
	fig.tight_layout()
	return _fig_to_b64(fig)


def generate_rfm_chart(rfm: pd.DataFrame) -> str | None:
	if rfm.empty or "Segment" not in rfm.columns:
		return None

	segment_order = ["Champions", "Loyal Customers", "At Risk", "Hibernating"]
	counts = rfm["Segment"].value_counts().reindex(segment_order).fillna(0)

	fig, ax = plt.subplots(figsize=(11, 6))
	bar_colors = ["#4338ca", "#5b4be0", "#7663ea", "#9f85f5"]
	bars = ax.bar(counts.index, counts.values, color=bar_colors, edgecolor="#312e81", linewidth=0.8)
	ax.set_title("RFM Segment Distribution", fontsize=14, fontweight="bold")
	ax.set_xlabel("Customer Segment", fontsize=11)
	ax.set_ylabel("Customers", fontsize=11)
	ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
	ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color=GRID_COLOR, alpha=0.9)
	ax.set_axisbelow(True)
	ax.tick_params(axis="x", rotation=15)

	for bar in bars:
		height = bar.get_height()
		ax.text(
			bar.get_x() + bar.get_width() / 2,
			height,
			f"{int(height):,}",
			ha="center",
			va="bottom",
			fontsize=9,
			color="#1f2937",
		)
	fig.tight_layout()
	return _fig_to_b64(fig)


def generate_cohort_chart(transactions: pd.DataFrame) -> str | None:
	if transactions.empty:
		return None

	cohort_df = transactions.copy()
	cohort_df = cohort_df.dropna(subset=["CustomerID", "InvoiceDate"])
	if cohort_df.empty:
		return None

	cohort_df["InvoiceMonth"] = cohort_df["InvoiceDate"].dt.to_period("M")
	cohort_df["CohortMonth"] = cohort_df.groupby("CustomerID")["InvoiceMonth"].transform("min")
	cohort_df["CohortIndex"] = (
		(cohort_df["InvoiceMonth"].dt.year - cohort_df["CohortMonth"].dt.year) * 12
		+ (cohort_df["InvoiceMonth"].dt.month - cohort_df["CohortMonth"].dt.month)
		+ 1
	)

	counts = (
		cohort_df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
		.nunique()
		.reset_index(name="Customers")
	)
	if counts.empty:
		return None

	pivot = counts.pivot(index="CohortMonth", columns="CohortIndex", values="Customers")
	if pivot.empty:
		return None

	sizes = pivot.iloc[:, 0]
	retention = (pivot.divide(sizes, axis=0) * 100).round(1)
	retention.index = retention.index.astype(str)

	fig, ax = plt.subplots(figsize=(12, 7))
	im = ax.imshow(retention.values, cmap="BuPu", aspect="auto")
	ax.set_title("Cohort Retention Heatmap", fontsize=14, fontweight="bold")
	ax.set_xlabel("Months Since First Purchase", fontsize=11)
	ax.set_ylabel("Cohort Month", fontsize=11)
	ax.set_xticks(range(retention.shape[1]))
	ax.set_xticklabels([str(c) for c in retention.columns])
	ax.set_yticks(range(retention.shape[0]))
	ax.set_yticklabels(retention.index)
	ax.set_xticks([x - 0.5 for x in range(1, retention.shape[1])], minor=True)
	ax.set_yticks([y - 0.5 for y in range(1, retention.shape[0])], minor=True)
	ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=0.6, alpha=0.85)
	ax.tick_params(axis="x", rotation=0)
	fig.colorbar(im, ax=ax, label="Retention %")
	fig.tight_layout()
	return _fig_to_b64(fig)


def get_segment_counts(rfm: pd.DataFrame) -> dict[str, int]:
	if rfm.empty or "Segment" not in rfm.columns:
		return {}

	segment_order = ["Champions", "Loyal Customers", "At Risk", "Hibernating"]
	counts = rfm["Segment"].value_counts().to_dict()
	return {segment: int(counts.get(segment, 0)) for segment in segment_order}


@app.route("/health")
def health():
	"""Health check endpoint - doesn't depend on data files"""
	return {
		"status": "ok",
		"app": "Infotact Analytics",
		"data_dir": str(DATA_DIR),
		"data_dir_exists": DATA_DIR.exists(),
		"transactions_file_exists": (DATA_DIR / TRANSACTIONS_FILE).exists(),
		"rfm_file_exists": (DATA_DIR / RFM_FILE).exists(),
	}, 200

@app.route("/")
def dashboard():
	try:
		all_transactions = clean_transactions(load_data(TRANSACTIONS_FILE))
		all_rfm = load_data(RFM_FILE)
		filters = parse_filters()

		available_countries: list[str] = []
		if not all_transactions.empty and "Country" in all_transactions.columns:
			available_countries = sorted(all_transactions["Country"].dropna().astype(str).unique().tolist())

		available_segments = ["Champions", "Loyal Customers", "At Risk", "Hibernating"]

		transactions, rfm = apply_filters(all_transactions, all_rfm, filters)
		kpis = calculate_kpis(transactions)
		sales_chart_image = generate_sales_chart(transactions)
		rfm_chart_image = generate_rfm_chart(rfm)
		segment_counts = get_segment_counts(rfm)
		cohort_chart_image = generate_cohort_chart(transactions)
		chart_version = "|".join([
			filters["start_date"],
			filters["end_date"],
			filters["segment"],
			filters["country"],
		])

		return render_template(
			"index.html",
			total_revenue=f"{kpis['total_revenue']:,.2f}",
			total_customers=f"{kpis['total_customers']:,}",
			average_order_value=f"{kpis['average_order_value']:,.2f}",
			sales_chart_image=sales_chart_image,
			rfm_chart_image=rfm_chart_image,
			segment_counts=segment_counts,
			cohort_chart_image=cohort_chart_image,
			chart_version=chart_version,
			available_segments=available_segments,
			available_countries=available_countries,
			selected_start_date=filters["start_date"],
			selected_end_date=filters["end_date"],
			selected_segment=filters["segment"],
			selected_country=filters["country"],
		)
	except Exception as e:
		print(f"[ERROR] Dashboard error: {e}")
		import traceback
		traceback.print_exc()
		return {
			"error": "Failed to load dashboard",
			"message": str(e),
			"data_dir": str(DATA_DIR),
			"data_dir_exists": DATA_DIR.exists(),
		}, 500


@app.route("/segments")
def segments_page():
	try:
		rfm = load_data(RFM_FILE)
		rfm_chart_image = generate_rfm_chart(rfm)
		segment_counts = get_segment_counts(rfm)
		rfm_sample = rfm.head(20).to_dict("records") if not rfm.empty else []

		return render_template(
			"rfm.html",
			segments=segment_counts,
			rfm_sample=rfm_sample,
			total_customers=len(rfm),
			rfm_chart_image=rfm_chart_image,
		)
	except Exception as e:
		print(f"[ERROR] Segments page error: {e}")
		import traceback
		traceback.print_exc()
		return {"error": "Failed to load segments", "message": str(e)}, 500


@app.route("/sales")
def sales_page():
	try:
		transactions = clean_transactions(load_data(TRANSACTIONS_FILE))
		sales_trend_image = generate_sales_chart(transactions)

		return render_template(
			"sales.html",
			sales_trend_image=sales_trend_image,
		)
	except Exception as e:
		print(f"[ERROR] Sales page error: {e}")
		import traceback
		traceback.print_exc()
		return {"error": "Failed to load sales", "message": str(e)}, 500


@app.route("/cohort")
def cohort_page():
	try:
		transactions = clean_transactions(load_data(TRANSACTIONS_FILE))
		cohort_chart_image = generate_cohort_chart(transactions)

		return render_template(
			"cohort.html",
			cohort_chart_image=cohort_chart_image,
		)
	except Exception as e:
		print(f"[ERROR] Cohort page error: {e}")
		import traceback
		traceback.print_exc()
		return {"error": "Failed to load cohort", "message": str(e)}, 500

@app.route("/test")
def test():
    return "App is working!"

if __name__ == "__main__":
	port = int(os.environ.get("PORT", 5000))
	app.run(host="0.0.0.0", port=port)
