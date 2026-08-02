import pandas as pd

df = pd.read_csv("event_graph/reports/eg_loso_v2_results.csv")

# We focus on Threshold = 0.5 for the contribution analysis
df_05 = df[df["Threshold"] == 0.5]

random_station_iol = df_05[df_05["split"] == "random_station"]["ERR_IoL"].values[0]

eg_loso = df_05[(df_05["split"] == "EG_LOSO")].copy()
eg_loso["Contribution"] = eg_loso["ERR_IoL"] - random_station_iol
eg_loso = eg_loso.sort_values(by="Contribution", ascending=False)

md = "# Station Contribution Analysis\n\n"
md += "This analysis measures the resilience of the network when a specific station is removed, compared to a random 20% station dropout. "
md += "A positive contribution means the network predicts this station better than it predicts a random subset (the station is highly redundant and well-covered by others). "
md += "A negative contribution means the station contains unique observations that the rest of the network struggles to reconstruct.\n\n"
md += f"**Baseline (Random Station 20% Drop) IoL:** {random_station_iol:.3f}\n\n"

md += "### Top Contributors (Highly Redundant / Well Covered)\n"
md += "| Station | IoL | Contribution |\n"
md += "| :--- | :--- | :--- |\n"
for _, row in eg_loso[eg_loso["Contribution"] > 0].iterrows():
    md += f"| {row['station']} | {row['ERR_IoL']:.3f} | +{row['Contribution']:.3f} |\n"

md += "\n### Weak Contributors (Unique / Poorly Covered)\n"
md += "| Station | IoL | Contribution |\n"
md += "| :--- | :--- | :--- |\n"
for _, row in eg_loso[eg_loso["Contribution"] <= 0].iterrows():
    md += f"| {row['station']} | {row['ERR_IoL']:.3f} | {row['Contribution']:.3f} |\n"

with open("docs/station_contribution_analysis.md", "w") as f:
    f.write(md)

print("Analysis written to docs/station_contribution_analysis.md")
