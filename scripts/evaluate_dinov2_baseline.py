import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "data" / "metadata"
EMBED = ROOT / "data" / "embeddings"
OUT = ROOT / "data" / "evaluation"

GOLD_FILE = EMBED / "dinov2_small_gold_embeddings.npz"
PROTO_FILE = EMBED / "dinov2_small_prototype_embeddings.npz"
LABEL_FILE = META / "verified_matching_pairs.csv"

TOP_K = 8


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm_name(value):
    return (value or "").strip().replace("\\", "/")


def load_embeddings(path):
    data = np.load(path, allow_pickle=False)
    return {
        "embeddings": data["embeddings"].astype(np.float32),
        "filenames": data["filenames"].astype(str),
        "sha256": data["sha256"].astype(str),
    }


def make_lookup(data):
    lookup = {}
    for i, filename in enumerate(data["filenames"]):
        lookup.setdefault(norm_name(filename), []).append(i)

    hash_lookup = {}
    for i, sha in enumerate(data["sha256"]):
        if sha:
            hash_lookup.setdefault(sha.strip(), []).append(i)

    return lookup, hash_lookup


def resolve_index(filename, sha, lookup, hash_lookup):
    name = norm_name(filename)
    if name in lookup and len(lookup[name]) == 1:
        return lookup[name][0]

    if sha:
        matches = hash_lookup.get(sha.strip(), [])
        if len(matches) == 1:
            return matches[0]

    return None


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    gold = load_embeddings(GOLD_FILE)
    proto = load_embeddings(PROTO_FILE)
    labels = read_csv(LABEL_FILE)

    if gold["embeddings"].shape[1] != proto["embeddings"].shape[1]:
        raise ValueError("Gold and prototype embedding dimensions differ.")

    gold_lookup, gold_hash_lookup = make_lookup(gold)
    proto_lookup, proto_hash_lookup = make_lookup(proto)

    # Embeddings are already L2-normalized, so dot product is cosine similarity.
    similarities = gold["embeddings"] @ proto["embeddings"].T

    report_rows = []
    mapped_labels = []
    unmapped = []

    for row in labels:
        label_text = (row.get("label") or "").strip()
        if label_text not in {"0", "1"}:
            continue

        gold_idx = resolve_index(
            row.get("gold_filename", ""),
            row.get("gold_sha256", ""),
            gold_lookup,
            gold_hash_lookup,
        )
        proto_idx = resolve_index(
            row.get("prototype_filename", ""),
            row.get("prototype_sha256", ""),
            proto_lookup,
            proto_hash_lookup,
        )

        if gold_idx is None or proto_idx is None:
            unmapped.append(
                {
                    "gold_filename": row.get("gold_filename", ""),
                    "prototype_filename": row.get("prototype_filename", ""),
                    "gold_mapped": str(gold_idx is not None),
                    "prototype_mapped": str(proto_idx is not None),
                    "label": label_text,
                }
            )
            continue

        mapped_labels.append(
            {
                "gold_idx": gold_idx,
                "proto_idx": proto_idx,
                "label": int(label_text),
                "gold_filename": gold["filenames"][gold_idx],
                "prototype_filename": proto["filenames"][proto_idx],
                "match_type": row.get("match_type", ""),
                "similarity": float(similarities[gold_idx, proto_idx]),
            }
        )

    # For every gold query, write its five highest-scoring prototypes.
    for gi, gold_filename in enumerate(gold["filenames"]):
        order = np.argsort(-similarities[gi])[:TOP_K]
        for rank, pi in enumerate(order, start=1):
            report_rows.append(
                {
                    "gold_filename": gold_filename,
                    "rank": rank,
                    "prototype_filename": proto["filenames"][pi],
                    "cosine_similarity": f"{similarities[gi, pi]:.6f}",
                }
            )

    with (OUT / "dinov2_top5_results.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "gold_filename",
                "rank",
                "prototype_filename",
                "cosine_similarity",
            ],
        )
        writer.writeheader()
        writer.writerows(report_rows)

    with (OUT / "dinov2_verified_pair_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = [
            "gold_filename",
            "prototype_filename",
            "label",
            "match_type",
            "cosine_similarity",
            "rank",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for item in mapped_labels:
            gi = item["gold_idx"]
            pi = item["proto_idx"]
            rank = int(np.sum(similarities[gi] > similarities[gi, pi])) + 1
            writer.writerow(
                {
                    "gold_filename": item["gold_filename"],
                    "prototype_filename": item["prototype_filename"],
                    "label": item["label"],
                    "match_type": item["match_type"],
                    "cosine_similarity": f'{item["similarity"]:.6f}',
                    "rank": rank,
                }
            )

    with (OUT / "dinov2_unmapped_labels.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        fields = [
            "gold_filename",
            "prototype_filename",
            "gold_mapped",
            "prototype_mapped",
            "label",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(unmapped)

    positives = [x for x in mapped_labels if x["label"] == 1]
    negatives = [x for x in mapped_labels if x["label"] == 0]

    # A query counts as a hit if its verified positive appears in the top 5.
    # Multiple verified positives for the same query are supported.
    positive_query_ids = set()
    hit_query_ids = set()

    for item in positives:
        gi = item["gold_idx"]
        positive_query_ids.add(gi)

        order = np.argsort(-similarities[gi])[:TOP_K]
        if item["proto_idx"] in order:
            hit_query_ids.add(gi)

    print("=" * 60)
    print("DINOV2 BASELINE EVALUATION")
    print("=" * 60)
    print("Gold queries:", len(gold["filenames"]))
    print("Prototype catalogue:", len(proto["filenames"]))
    print("Label rows read:", len(labels))
    print("Mapped labelled pairs:", len(mapped_labels))
    print("Positive pairs:", len(positives))
    print("Negative pairs:", len(negatives))
    print("Unmapped label rows:", len(unmapped))

    if positive_query_ids:
        recall_at_5 = len(hit_query_ids) / len(positive_query_ids)
        print(
            f"Positive-query Recall@5: {len(hit_query_ids)}/"
            f"{len(positive_query_ids)} = {recall_at_5:.3f}"
        )
    else:
        print("Positive-query Recall@5: unavailable; no mapped positives.")

    if positives:
        ranks = []
        for item in positives:
            gi = item["gold_idx"]
            pi = item["proto_idx"]
            ranks.append(int(np.sum(similarities[gi] > similarities[gi, pi])) + 1)

        pair_recall_at_5 = sum(rank <= TOP_K for rank in ranks) / len(ranks)
        median_rank = float(np.median(ranks))
        print(
            f"Positive-pair Recall@5: "
            f"{sum(rank <= TOP_K for rank in ranks)}/{len(ranks)}"
            f" = {pair_recall_at_5:.3f}"
        )
        print(f"Median rank of verified positive pairs: {median_rank:.1f}")

    if negatives:
        negative_ranks = []
        for item in negatives:
            gi = item["gold_idx"]
            pi = item["proto_idx"]
            negative_ranks.append(
                int(np.sum(similarities[gi] > similarities[gi, pi])) + 1
            )
        print(
            "Negative pairs ranked in top 5:",
            sum(rank <= TOP_K for rank in negative_ranks),
            "of",
            len(negative_ranks),
        )

    print("\nReports saved to:", OUT)
    print("No model or MongoDB records were changed.")
    print(
        "\nCaution: these are preliminary results. Multiple photos of one "
        "design, incomplete labels, and duplicate designs can affect metrics."
    )


if __name__ == "__main__":
    main()
