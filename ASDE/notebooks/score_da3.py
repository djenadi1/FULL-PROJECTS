import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_percentage_error

index = json.load(open("../data/index.json"))
rows = []

for item in index:
    if item["split"] == "test":
        continue

    name = item["name"]
    exact = np.load(f"../data/frames/{name}_depth.npy")
    da3 = np.load(f"../DA3_OUT/{name}_da3.npy")

    # sky is 0 in the exact depth, so leave it out
    keep = exact > 0

    rows.append({
        "weather": item["weather"],
        "distance_error": mean_absolute_percentage_error(exact[keep], da3[keep]),
        "near_far_order": spearmanr(exact[keep][::7], da3[keep][::7]).statistic,
    })

scores = pd.DataFrame(rows)
print("overall")
print(scores[["distance_error", "near_far_order"]].mean())
print()
print(scores.groupby("weather")[["distance_error", "near_far_order"]].mean())
