from pathlib import Path

import pandas as pd


def clean_transactions(df: pd.DataFrame) -> pd.DataFrame:
	cleaned = df.copy()

	cleaned = cleaned.dropna(subset=["CustomerID"])
	cleaned = cleaned[(cleaned["Quantity"] > 0) & (cleaned["Price"] > 0)]
	cleaned["InvoiceDate"] = pd.to_datetime(cleaned["InvoiceDate"], errors="coerce")
	cleaned = cleaned.dropna(subset=["InvoiceDate"])
	cleaned["TotalAmount"] = cleaned["Quantity"] * cleaned["Price"]

	return cleaned


def run_sql_like_transformations(cleaned: pd.DataFrame) -> None:
	# Country-level sales summary.
	country_sales = (
		cleaned.groupby("Country", as_index=False)
		.agg(
			TotalSales=("TotalAmount", "sum"),
			Transactions=("InvoiceID", "count"),
			UniqueCustomers=("CustomerID", "nunique"),
		)
		.sort_values("TotalSales", ascending=False)
	)

	# High-value orders summarized by product.
	high_value = cleaned[cleaned["TotalAmount"] >= 500]
	top_products_high_value = (
		high_value.groupby("Product", as_index=False)
		.agg(
			HighValueOrders=("InvoiceID", "count"),
			HighValueSales=("TotalAmount", "sum"),
		)
		.sort_values("HighValueSales", ascending=False)
	)

	print("Top 10 countries by total sales:")
	print(country_sales.head(10).to_string(index=False))
	print("\nTop 10 products in high-value orders (TotalAmount >= 500):")
	print(top_products_high_value.head(10).to_string(index=False))


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	input_path = project_root / "data" / "transactions.csv"
	output_path = project_root / "data" / "cleaned_transactions.csv"

	df = pd.read_csv(input_path)
	cleaned_df = clean_transactions(df)
	cleaned_df.to_csv(output_path, index=False)

	print(f"Input rows: {len(df)}")
	print(f"Cleaned rows: {len(cleaned_df)}")
	print(f"Saved cleaned dataset to: {output_path}")

	run_sql_like_transformations(cleaned_df)


if __name__ == "__main__":
	main()
