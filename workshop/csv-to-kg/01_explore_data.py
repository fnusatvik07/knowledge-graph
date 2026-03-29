"""Hour 3 — Script 01: Explore the AI Companies Dataset"""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).parent

df = pd.read_csv(HERE / "data" / "ai_companies.csv")

print(f"Dataset: {df.shape[0]} companies, {df.shape[1]} columns\n")
print(f"Columns: {', '.join(df.columns)}\n")
print(df[["company", "ceo", "category", "headquarters"]].to_string(index=False))

print(f"\n\nCategories: {df['category'].nunique()}")
for cat, count in df['category'].value_counts().items():
    print(f"  {cat}: {count}")

# Count unique investors across all companies
all_investors = set()
for inv_str in df["investors"].dropna():
    for inv in inv_str.split(";"):
        all_investors.add(inv.strip())

print(f"\nUnique investors: {len(all_investors)}")
print(f"Cities: {df['headquarters'].nunique()}")
print(f"Total funding: ${df['funding_total_b'].sum():.1f}B")

print("\n" + "=" * 50)
print("This flat CSV has HIDDEN RELATIONSHIPS:")
print("  - Investors fund MULTIPLE companies")
print("  - Companies share TECHNOLOGIES")
print("  - CEOs lead companies in the SAME city")
print("A Knowledge Graph will reveal these connections!")
