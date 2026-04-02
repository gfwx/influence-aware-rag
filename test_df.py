import pandas as pd
results = []
summary = pd.DataFrame([
    {
        "Query ID": r["query_id"],
        "Divergent": r["spearman_rho"] < 0.7
    }
    for r in results
])
print(summary)
