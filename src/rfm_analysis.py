"""RFM segmentation pipeline for retail customers."""

import logging
from pathlib import Path

import pandas as pd


logger = logging.getLogger(__name__)
REQUIRED_COLUMNS = {"InvoiceDate", "CustomerID", "InvoiceID", "TotalAmount"}


def get_paths() -> tuple[Path, Path]:
	"""Return input and output paths for the RFM job."""
	project_root = Path(__file__).resolve().parent.parent
	input_path = project_root / "data" / "cleaned_transactions.csv"
	output_path = project_root / "data" / "rfm_segments.csv"
	return input_path, output_path


def load_data(input_path: Path) -> pd.DataFrame:
	"""Load cleaned transaction data from disk."""
	return pd.read_csv(input_path)


def validate_input(df: pd.DataFrame) -> None:
	"""Validate required columns before processing."""
	missing_columns = REQUIRED_COLUMNS.difference(df.columns)
	if missing_columns:
		missing = ", ".join(sorted(missing_columns))
		raise ValueError(f"Missing required columns: {missing}")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
	"""Normalize transaction data used for RFM scoring."""
	validate_input(df)
	cleaned = df.copy()
	cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
	return cleaned.dropna(subset=["InvoiceDate", "CustomerID"])


def _quantile_score(series: pd.Series, reverse: bool = False) -> pd.Series:
	"""Convert a numeric series into 1-5 quantile-based scores."""
	labels = [1, 2, 3, 4, 5]
	# Rank first to keep qcut stable when many values are tied.
	ranked = series.rank(method="first")
	scores = pd.qcut(ranked, q=5, labels=labels).astype(int)
	if reverse:
		return 6 - scores
	return scores


def assign_customer_segment(row: pd.Series) -> str:
	"""Map RFM scores to a customer segment label."""
	recency_score = row["R_Score"]
	frequency_score = row["F_Score"]
	monetary_score = row["M_Score"]

	if recency_score >= 4 and frequency_score >= 4 and monetary_score >= 4:
		return "Champions"
	if recency_score >= 3 and frequency_score >= 3 and monetary_score >= 3:
		return "Loyal Customers"
	if recency_score <= 2 and (frequency_score >= 3 or monetary_score >= 3):
		return "At Risk"
	return "Hibernating"


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
	"""Calculate RFM metrics, scores, and segment labels per customer."""
	df = clean_data(df)
	if df.empty:
		return pd.DataFrame(
			columns=[
				"CustomerID",
				"Recency",
				"Frequency",
				"Monetary",
				"R_Score",
				"F_Score",
				"M_Score",
				"RFM_Score",
				"Segment",
			]
		)

	reference_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

	rfm = (
		df.groupby("CustomerID", as_index=False)
		.agg(
			Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
			Frequency=("InvoiceID", "count"),
			Monetary=("TotalAmount", "sum"),
		)
	)

	rfm["R_Score"] = _quantile_score(rfm["Recency"], reverse=True)
	rfm["F_Score"] = _quantile_score(rfm["Frequency"])
	rfm["M_Score"] = _quantile_score(rfm["Monetary"])
	rfm["RFM_Score"] = (
		rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
	)
	rfm["Segment"] = rfm.apply(assign_customer_segment, axis=1)

	return rfm.sort_values(["R_Score", "F_Score", "M_Score"], ascending=False)


def save_output(rfm_df: pd.DataFrame, output_path: Path) -> None:
	"""Persist RFM results to CSV."""
	output_path.parent.mkdir(parents=True, exist_ok=True)
	rfm_df.to_csv(output_path, index=False)


def main() -> None:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
	input_path, output_path = get_paths()

	df = load_data(input_path)
	rfm_df = calculate_rfm(df)
	save_output(rfm_df, output_path)

	logger.info("Customers scored: %s", len(rfm_df))
	logger.info("Saved RFM segments to: %s", output_path)
	if not rfm_df.empty:
		logger.info("Segment distribution:\n%s", rfm_df["Segment"].value_counts().to_string())


if __name__ == "__main__":
	main()
