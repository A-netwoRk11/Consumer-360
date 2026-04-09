from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def build_cohort_matrix(df: pd.DataFrame) -> pd.DataFrame:
	df = df.copy()
	df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
	df = df.dropna(subset=["CustomerID", "InvoiceDate"])

	df["InvoiceMonth"] = df["InvoiceDate"].dt.to_period("M")
	df["CohortMonth"] = df.groupby("CustomerID")["InvoiceMonth"].transform("min")

	df["CohortIndex"] = (
		(df["InvoiceMonth"].dt.year - df["CohortMonth"].dt.year) * 12
		+ (df["InvoiceMonth"].dt.month - df["CohortMonth"].dt.month)
		+ 1
	)

	cohort_counts = (
		df.groupby(["CohortMonth", "CohortIndex"])["CustomerID"]
		.nunique()
		.reset_index(name="Customers")
	)

	cohort_pivot = cohort_counts.pivot(index="CohortMonth", columns="CohortIndex", values="Customers")
	cohort_sizes = cohort_pivot.iloc[:, 0]
	retention = cohort_pivot.divide(cohort_sizes, axis=0) * 100

	retention.index = retention.index.astype(str)
	retention = retention.round(2)

	return retention


def plot_heatmap(retention: pd.DataFrame, output_path: Path) -> None:
	fig, ax = plt.subplots(figsize=(12, 7))

	data = retention.values
	im = ax.imshow(data, cmap="YlGnBu", aspect="auto")

	ax.set_title("Customer Cohort Retention Heatmap (%)", fontsize=14, pad=12)
	ax.set_xlabel("Months Since First Purchase")
	ax.set_ylabel("Cohort Month")
	ax.set_xticks(np.arange(retention.shape[1]))
	ax.set_xticklabels(retention.columns.astype(str))
	ax.set_yticks(np.arange(retention.shape[0]))
	ax.set_yticklabels(retention.index)

	# Show each retention value inside its cell.
	for i in range(retention.shape[0]):
		for j in range(retention.shape[1]):
			value = retention.iat[i, j]
			if pd.notna(value):
				ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="black", fontsize=8)

	fig.colorbar(im, ax=ax, label="Retention %")
	fig.tight_layout()
	fig.savefig(output_path, dpi=150)
	plt.close(fig)


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	input_path = project_root / "data" / "cleaned_transactions.csv"
	retention_csv_path = project_root / "data" / "cohort_retention.csv"
	heatmap_path = project_root / "data" / "cohort_retention_heatmap.png"

	df = pd.read_csv(input_path)
	retention = build_cohort_matrix(df)

	retention.to_csv(retention_csv_path)
	plot_heatmap(retention, heatmap_path)

	print(f"Saved cohort retention table to: {retention_csv_path}")
	print(f"Saved cohort heatmap to: {heatmap_path}")
	print("\nRetention matrix preview:")
	print(retention.head().to_string())


if __name__ == "__main__":
	main()
