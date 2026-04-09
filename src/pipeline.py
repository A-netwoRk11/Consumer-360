from pathlib import Path
import logging

import pandas as pd

from generate_data import generate_transactions
from data_cleaning import clean_transactions
from rfm_analysis import calculate_rfm
from cohort_analysis import build_cohort_matrix, plot_heatmap
from market_basket import build_basket, generate_rules


logger = logging.getLogger("retail_pipeline")


def setup_logging() -> None:
	logging.basicConfig(
		level=logging.INFO,
		format="%(asctime)s | %(levelname)s | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)


def get_paths() -> dict[str, Path]:
	project_root = Path(__file__).resolve().parent.parent
	data_dir = project_root / "data"
	data_dir.mkdir(parents=True, exist_ok=True)

	return {
		"transactions": data_dir / "transactions.csv",
		"cleaned": data_dir / "cleaned_transactions.csv",
		"rfm": data_dir / "rfm_segments.csv",
		"cohort": data_dir / "cohort_retention.csv",
		"cohort_heatmap": data_dir / "cohort_retention_heatmap.png",
		"rules": data_dir / "association_rules.csv",
	}


def step_generate_data(output_path: Path, num_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
	logger.info("[1/5] Generating synthetic retail transaction data...")
	df = generate_transactions(num_rows=num_rows, seed=seed)
	df.to_csv(output_path, index=False)
	logger.info("Generated %s rows -> %s", len(df), output_path)
	return df


def step_clean_data(input_path: Path, output_path: Path) -> pd.DataFrame:
	logger.info("[2/5] Cleaning transaction data...")
	df = pd.read_csv(input_path)
	cleaned = clean_transactions(df)
	cleaned.to_csv(output_path, index=False)
	logger.info("Cleaned rows: %s (from %s) -> %s", len(cleaned), len(df), output_path)
	return cleaned


def step_rfm_analysis(input_path: Path, output_path: Path) -> pd.DataFrame:
	logger.info("[3/5] Running RFM analysis...")
	cleaned = pd.read_csv(input_path)
	rfm = calculate_rfm(cleaned)
	rfm.to_csv(output_path, index=False)
	logger.info("Scored customers: %s -> %s", len(rfm), output_path)
	return rfm


def step_cohort_analysis(input_path: Path, csv_path: Path, heatmap_path: Path) -> pd.DataFrame:
	logger.info("[4/5] Running cohort analysis...")
	cleaned = pd.read_csv(input_path)
	retention = build_cohort_matrix(cleaned)
	retention.to_csv(csv_path)
	plot_heatmap(retention, heatmap_path)
	logger.info("Saved cohort table -> %s", csv_path)
	logger.info("Saved cohort heatmap -> %s", heatmap_path)
	return retention


def step_market_basket_analysis(input_path: Path, output_path: Path) -> pd.DataFrame:
	logger.info("[5/5] Running market basket analysis (Apriori + association rules)...")
	df = pd.read_csv(input_path)
	df = df.dropna(subset=["InvoiceID", "CustomerID", "Product", "InvoiceDate"])
	df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
	df = df.dropna(subset=["InvoiceDate"])

	baskets: list[tuple[str, pd.DataFrame]] = []
	baskets.append(("InvoiceID", build_basket(df, "InvoiceID")))
	df["BasketKey"] = df["CustomerID"].astype(str) + "-" + df["InvoiceDate"].dt.to_period("M").astype(str)
	baskets.append(("CustomerID-Month", build_basket(df, "BasketKey")))
	baskets.append(("CustomerID", build_basket(df, "CustomerID")))

	basket_type = "InvoiceID"
	rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

	for name, basket in baskets:
		_, candidate_rules = generate_rules(basket)
		basket_type = name
		rules = candidate_rules
		if not rules.empty:
			break

	rules.to_csv(output_path, index=False)
	logger.info("Basket granularity used: %s", basket_type)
	logger.info("Association rules found: %s -> %s", len(rules), output_path)
	return rules


def run_pipeline() -> None:
	paths = get_paths()
	step_generate_data(paths["transactions"])
	step_clean_data(paths["transactions"], paths["cleaned"])
	step_rfm_analysis(paths["cleaned"], paths["rfm"])
	step_cohort_analysis(paths["cleaned"], paths["cohort"], paths["cohort_heatmap"])
	step_market_basket_analysis(paths["cleaned"], paths["rules"])
	logger.info("Pipeline completed successfully.")


def main() -> None:
	setup_logging()
	run_pipeline()


if __name__ == "__main__":
	main()
