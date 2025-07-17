import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from datetime import datetime

# Create timestamp for filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")



# ----------------------
# CONFIG
# ----------------------

ABLATION_PATHS = {
    "key_original": "models/faiss_indices/key_original.index",
    "key_nlp": "models/faiss_indices/key_nlp.index",
    "key_llm": "models/faiss_indices/key_llm.index",
    "desc_original": "models/faiss_indices/desc_original.index",
    "desc_llm": "models/faiss_indices/desc_llm.index",
    "full_OG": "models/faiss_indices/full_OG.index",
    "full_nlp": "models/faiss_indices/full_nlp.index",
    "full_llm": "models/faiss_indices/full_llm.index",
    "onlykey_original": "models/faiss_indices/onlykey_original.index",
    "onlykey_nlp": "models/faiss_indices/onlykey_nlp.index",
    "onlykey_llm": "models/faiss_indices/onlykey_llm.index",
    "onlytopic_original": "models/faiss_indices/onlytopic_original.index",
    "onlytopic_nlp": "models/faiss_indices/onlytopic_nlp.index",
    "onlytopic_llm": "models/faiss_indices/onlytopic_llm.index",
    "onlytitle": "models/faiss_indices/onlytitle.index"
}


ID_PATHS = {
    name: path.replace(".index", "_ids.json") for name, path in ABLATION_PATHS.items()
}

QUERY_PATH = "data/user_queries1.json"
EMBEDDING_MODEL = "BAAI/bge-base-en"
TOP_K = 5
PLOT_SAVE_PATH = f"evaluation/faiss_evaluation_plot_{timestamp}.png"


# ----------------------
# LOAD MODEL & QUERIES
# ----------------------

print(f"🔍 Loading embedding model: {EMBEDDING_MODEL}")
model = SentenceTransformer(EMBEDDING_MODEL)

with open(QUERY_PATH, "r", encoding="utf-8") as f:
    queries = json.load(f)

# ----------------------
# EVALUATION
# ----------------------

results = defaultdict(lambda: {"hit@1": 0, "hit@3": 0, "hit@5": 0, "rr_sum": 0, "total": 0})
similarity_stats = defaultdict(lambda: {"correct_similarities": [], "topk_similarities": []})

# NEW: Track by query type too
results_by_query_type = defaultdict(lambda: defaultdict(lambda: {
    "hit@1": 0, "hit@3": 0, "hit@5": 0, "rr_sum": 0, "total": 0
}))

for ablation_name, index_path in ABLATION_PATHS.items():
    print(f"\n📊 Evaluating ablation: {ablation_name}")
    index = faiss.read_index(index_path)

    with open(ID_PATHS[ablation_name], "r", encoding="utf-8") as f:
        id_list = json.load(f)

    for entry in tqdm(queries, desc=f"🔄 Running queries for {ablation_name}"):
        target_id = entry["dataset_id"]
        for query_type, query_text in entry["queries"].items():
            embedding = model.encode(
                f"Represent this query for retrieval: {query_text}",
                normalize_embeddings=False
            )

            D, I = index.search(np.array([embedding]), TOP_K)
            retrieved_ids = [id_list[i] for i in I[0]]
                # Debug print for onlytopic ablations
            if "onlytopic" in ablation_name:
                print(f"\n[DEBUG] Ablation: {ablation_name}")
                print(f"Query Type: {query_type}")
                print(f"Query Text: {query_text}")
                print(f"Target ID: {target_id}")
                print(f"Top-5 Retrieved IDs: {retrieved_ids}")  
            # Overall stats
            results[ablation_name]["total"] += 1
            if target_id in retrieved_ids:
                rank = retrieved_ids.index(target_id)
                if rank == 0:
                    results[ablation_name]["hit@1"] += 1
                if rank < 3:
                    results[ablation_name]["hit@3"] += 1
                if rank < 5:
                    results[ablation_name]["hit@5"] += 1
                results[ablation_name]["rr_sum"] += 1 / (rank + 1)

            # Query-type-specific stats
            results_by_query_type[ablation_name][query_type]["total"] += 1
            if target_id in retrieved_ids:
                if rank == 0:
                    results_by_query_type[ablation_name][query_type]["hit@1"] += 1
                if rank < 3:
                    results_by_query_type[ablation_name][query_type]["hit@3"] += 1
                if rank < 5:
                    results_by_query_type[ablation_name][query_type]["hit@5"] += 1
                results_by_query_type[ablation_name][query_type]["rr_sum"] += 1 / (rank + 1)

            # Similarities
            dot_sims = D[0]
            similarity_stats[ablation_name]["topk_similarities"].extend(dot_sims.tolist())
            if target_id in retrieved_ids:
                similarity_stats[ablation_name]["correct_similarities"].append(
                    dot_sims[retrieved_ids.index(target_id)]
                )


# ----------------------
# PRINT SUMMARY
# ----------------------

print("\n📈 Evaluation Results:")
print(f"{'Ablation':<20} {'Hit@1':<8} {'Hit@3':<8} {'Hit@5':<8} {'MRR':<8}")
print("-" * 60)

summary_data = []
for ablation, score in results.items():
    total = score["total"] or 1
    hit1 = score["hit@1"] / total
    hit3 = score["hit@3"] / total
    hit5 = score["hit@5"] / total
    mrr = score["rr_sum"] / total
    print(f"{ablation:<20} {hit1:<8.3f} {hit3:<8.3f} {hit5:<8.3f} {mrr:<8.3f}")
    summary_data.append({
        "Ablation": ablation,
        "Hit@1": hit1,
        "Hit@3": hit3,
        "Hit@5": hit5,
        "MRR": mrr
    })

# ----------------------
# PLOT OVERALL RESULTS
# ----------------------

df = pd.DataFrame(summary_data).set_index("Ablation")
colors = sns.color_palette("Set2", n_colors=len(df.columns))

ax = df.plot(kind="bar", figsize=(12, 7), rot=30, width=0.75, color=colors, edgecolor='black')
for container in ax.containers:
    ax.bar_label(container, fmt='%.2f', label_type='edge', fontsize=9, padding=3)

plt.title("📊 FAISS Retrieval Performance by Ablation", fontsize=14)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1.05)
plt.grid(axis="y", linestyle="--", linewidth=0.5)
plt.legend(loc="upper right", frameon=True)
plt.tight_layout()
plt.savefig(PLOT_SAVE_PATH)
print(f"\n✅ Saved plot to: {PLOT_SAVE_PATH}")

# ----------------------
# PLOT VIOLIN PLOT
# ----------------------

plot_data = []
for ablation, sims in similarity_stats.items():
    plot_data.extend([
        {"Ablation": ablation, "Type": "Correct", "Similarity": sim}
        for sim in sims["correct_similarities"]
    ])
    plot_data.extend([
        {"Ablation": ablation, "Type": "Top-k", "Similarity": sim}
        for sim in sims["topk_similarities"]
    ])

df_sim = pd.DataFrame(plot_data)
plt.figure(figsize=(14, 7))
sns.violinplot(data=df_sim, x="Ablation", y="Similarity", hue="Type", split=True,
               inner="quart", linewidth=1.2)
plt.title("Dot Product Similarity Distributions by Ablation", fontsize=14)
plt.xlabel("Ablation", fontsize=12)
plt.ylabel("Dot Product", fontsize=12)
plt.xticks(rotation=30)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"evaluation/similarity_violin_plot_{timestamp}.png")
plt.close()
print("✅ Saved violin plot to: evaluation/similarity_violin_plot.png")

# ----------------------
# GRANULAR PLOT BY QUERY TYPE — One Plot per Query Type
# ----------------------

granular_data = []
for ablation, query_types in results_by_query_type.items():
    for qtype, score in query_types.items():
        total = score["total"] or 1
        granular_data.append({
            "Ablation": ablation,
            "QueryType": qtype,
            "Hit@1": score["hit@1"] / total,
            "Hit@3": score["hit@3"] / total,
            "Hit@5": score["hit@5"] / total,
            "MRR": score["rr_sum"] / total
        })

df_granular = pd.DataFrame(granular_data)
# Optional: Print the granular results table
print("\n📋 Query-Type Breakdown:")
print(df_granular.round(3).to_string(index=False))

# Optional: Save the raw numbers to CSV for analysis
csv_path = f"evaluation/query_type_metrics_{timestamp}.csv"
df_granular.to_csv(csv_path, index=False)
print(f"📄 Saved query-type metrics to: {csv_path}")

melted = df_granular.melt(id_vars=["Ablation", "QueryType"], 
                          value_vars=["Hit@1", "Hit@3", "Hit@5", "MRR"], 
                          var_name="Metric", value_name="Score")

g = sns.catplot(
    data=melted,
    kind="bar",
    x="Ablation",
    y="Score",
    hue="Metric",
    col="QueryType",
    col_wrap=3,
    height=5,
    aspect=1.3,
    palette="Set2"
)

g.fig.subplots_adjust(top=0.9)
g.fig.suptitle("📊 Score Breakdown by Ablation for Each Query Type", fontsize=16)

for ax in g.axes.flatten():
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
g.savefig(f"evaluation/query_type_breakdown_{timestamp}.png")
print(f"✅ Saved query-type breakdown plot to: evaluation/query_type_breakdown_{timestamp}.png")
