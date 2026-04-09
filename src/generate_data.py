from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_transactions(num_rows: int = 6000, seed: int = 42) -> pd.DataFrame:
	rng = np.random.default_rng(seed)

	products = {
		"Running Shoes": (55, 180),
		"Casual Sneakers": (45, 140),
		"Formal Shirt": (20, 80),
		"T-Shirt": (8, 35),
		"Jeans": (25, 95),
		"Jacket": (60, 220),
		"Bluetooth Headphones": (35, 160),
		"Smartphone": (250, 1200),
		"Laptop": (500, 2500),
		"Wireless Mouse": (12, 65),
		"Coffee Maker": (30, 180),
		"Backpack": (18, 90),
		"Sunglasses": (15, 120),
		"Sports Watch": (80, 450),
		"Desk Lamp": (18, 75),
	}

	countries = [
		"United Kingdom",
		"United States",
		"Germany",
		"France",
		"Netherlands",
		"Spain",
		"Italy",
		"Canada",
		"Australia",
		"India",
	]

	country_probs = np.array([0.20, 0.18, 0.11, 0.08, 0.07, 0.07, 0.06, 0.08, 0.06, 0.09])

	product_names = np.array(list(products.keys()))
	product_probs = np.array([0.10, 0.09, 0.08, 0.11, 0.09, 0.07, 0.08, 0.06, 0.05, 0.06, 0.05, 0.06, 0.03, 0.04, 0.03])

	chosen_products = rng.choice(product_names, size=num_rows, p=product_probs)

	invoice_ids = [f"INV-{100000 + i}" for i in range(num_rows)]
	customer_ids = rng.integers(10000, 99999, size=num_rows).astype(object)

	quantities = rng.integers(1, 8, size=num_rows)
	prices = np.empty(num_rows, dtype=float)
	for i, product in enumerate(chosen_products):
		low, high = products[product]
		prices[i] = round(rng.uniform(low, high), 2)

	end_date = datetime.now()
	start_date = end_date - timedelta(days=730)
	total_seconds = int((end_date - start_date).total_seconds())
	random_seconds = rng.integers(0, total_seconds, size=num_rows)
	invoice_dates = [start_date + timedelta(seconds=int(s)) for s in random_seconds]

	selected_countries = rng.choice(countries, size=num_rows, p=country_probs)

	df = pd.DataFrame(
		{
			"InvoiceID": invoice_ids,
			"CustomerID": customer_ids,
			"Product": chosen_products,
			"Quantity": quantities,
			"Price": prices,
			"InvoiceDate": pd.to_datetime(invoice_dates),
			"Country": selected_countries,
		}
	)

	# Add a small amount of messy data so cleaning logic has something realistic to handle.
	null_customer_count = max(1, int(num_rows * 0.03))
	negative_quantity_count = max(1, int(num_rows * 0.015))
	negative_price_count = max(1, int(num_rows * 0.01))

	null_idx = rng.choice(df.index, size=null_customer_count, replace=False)
	remaining_idx = np.setdiff1d(df.index, null_idx)
	neg_qty_idx = rng.choice(remaining_idx, size=negative_quantity_count, replace=False)
	remaining_idx = np.setdiff1d(remaining_idx, neg_qty_idx)
	neg_price_idx = rng.choice(remaining_idx, size=negative_price_count, replace=False)

	df.loc[null_idx, "CustomerID"] = np.nan
	df.loc[neg_qty_idx, "Quantity"] = -rng.integers(1, 4, size=negative_quantity_count)
	df.loc[neg_price_idx, "Price"] = -np.round(df.loc[neg_price_idx, "Price"].values, 2)

	return df


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	output_path = project_root / "data" / "transactions.csv"
	output_path.parent.mkdir(parents=True, exist_ok=True)

	df = generate_transactions(num_rows=6000, seed=42)
	df.to_csv(output_path, index=False)
	print(f"Generated {len(df)} rows at {output_path}")


if __name__ == "__main__":
	main()
