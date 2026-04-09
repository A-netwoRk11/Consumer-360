from pathlib import Path

import pandas as pd

try:
	from mlxtend.frequent_patterns import apriori, association_rules
except ImportError as exc:
	raise ImportError(
		"mlxtend is required. Install it with: pip install mlxtend"
	) from exc


def build_basket(df: pd.DataFrame, key_col: str) -> pd.DataFrame:
	basket = (
		df.groupby([key_col, "Product"])
		.size()
		.unstack(fill_value=0)
	)
	return (basket > 0)


def generate_rules(basket: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
	frequent_itemsets = apriori(basket, min_support=0.002, use_colnames=True)
	if frequent_itemsets.empty:
		return frequent_itemsets, pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

	rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.05)
	if rules.empty:
		return frequent_itemsets, pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

	rules = rules[rules["lift"] >= 1.0]
	if rules.empty:
		return frequent_itemsets, pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

	rules = rules[
		[
			"antecedents",
			"consequents",
			"support",
			"confidence",
			"lift",
		]
	].copy()
	rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(list(x))))
	rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(list(x))))
	rules = rules.sort_values(["lift", "confidence", "support"], ascending=False)

	return frequent_itemsets, rules


def main() -> None:
	project_root = Path(__file__).resolve().parent.parent
	input_path = project_root / "data" / "cleaned_transactions.csv"
	output_rules = project_root / "data" / "association_rules.csv"

	df = pd.read_csv(input_path)
	df = df.dropna(subset=["InvoiceID", "CustomerID", "Product", "InvoiceDate"])
	df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
	df = df.dropna(subset=["InvoiceDate"])

	baskets: list[tuple[str, pd.DataFrame]] = []
	baskets.append(("InvoiceID", build_basket(df, "InvoiceID")))

	# Fallback when most invoices contain only one product.
	df["BasketKey"] = (
		df["CustomerID"].astype(str)
		+ "-"
		+ df["InvoiceDate"].dt.to_period("M").astype(str)
	)
	baskets.append(("CustomerID-Month", build_basket(df, "BasketKey")))
	baskets.append(("CustomerID", build_basket(df, "CustomerID")))

	basket_type = "InvoiceID"
	used_basket = baskets[0][1]
	frequent_itemsets = pd.DataFrame()
	rules = pd.DataFrame(columns=["antecedents", "consequents", "support", "confidence", "lift"])

	for name, basket in baskets:
		fi, rl = generate_rules(basket)
		basket_type = name
		used_basket = basket
		frequent_itemsets = fi
		rules = rl
		if not rules.empty:
			break

	rules.to_csv(output_rules, index=False)

	print(f"Basket type used: {basket_type}")
	print(f"Baskets analyzed: {used_basket.shape[0]}")
	print(f"Frequent itemsets found: {len(frequent_itemsets)}")
	print(f"Association rules found: {len(rules)}")
	print(f"Saved rules to: {output_rules}")

	if not rules.empty:
		print("\nTop 10 rules by lift:")
		print(rules.head(10).to_string(index=False))


if __name__ == "__main__":
	main()
