import numpy as np


def unit_vector(vector):
    """Normalize a vector safely."""
    vector = np.asarray(vector, dtype=np.float32)
    norm = np.linalg.norm(vector)

    if norm < 1e-12:
        return np.zeros_like(vector)

    return vector / norm


def global_similarity(query_global, prototype_globals):
    """
    Compare a query global embedding against multiple prototype views.

    query_global: (384,)
    prototype_globals: (number_of_views, 384)
    """
    q = unit_vector(query_global)
    p = np.asarray(prototype_globals, dtype=np.float32)

    if p.size == 0 or q.ndim != 1:
        return -1.0

    if p.ndim == 1:
        p = p.reshape(1, -1)

    if p.shape[1] != q.shape[0]:
        return -1.0

    norms = np.linalg.norm(p, axis=1, keepdims=True)
    p = p / np.maximum(norms, 1e-12)

    return float(np.max(p @ q))


def local_patch_similarity(query_patches, prototype_patches, fraction=0.20):
    """
    Compare local patch embeddings.

    Each input should have shape (number_of_patches, embedding_dimension).
    Uses bidirectional nearest-patch similarity.
    """
    q = np.asarray(query_patches, dtype=np.float32)
    p = np.asarray(prototype_patches, dtype=np.float32)

    if q.ndim != 2 or p.ndim != 2:
        return -1.0

    if q.size == 0 or p.size == 0:
        return -1.0

    if q.shape[1] != p.shape[1]:
        return -1.0

    q_norms = np.linalg.norm(q, axis=1, keepdims=True)
    p_norms = np.linalg.norm(p, axis=1, keepdims=True)

    q = q / np.maximum(q_norms, 1e-12)
    p = p / np.maximum(p_norms, 1e-12)

    # Similarity matrix: query patches x prototype patches.
    similarity = q @ p.T

    query_to_prototype = np.max(similarity, axis=1)
    prototype_to_query = np.max(similarity, axis=0)

    def top_mean(values):
        count = max(1, int(np.ceil(len(values) * fraction)))
        count = min(count, len(values))

        top_values = np.partition(values, len(values) - count)[-count:]

        return float(np.mean(top_values))

    score_forward = top_mean(query_to_prototype)
    score_reverse = top_mean(prototype_to_query)

    return (score_forward + score_reverse) / 2.0


def combined_similarity(query_globals, query_patches, prototype_document):
    """
    Combine global and local similarity across multiple views.

    Expected prototype fields:
      global_embeddings: (views, 384)
      patch_embeddings: (views, patches, 384)

    Falls back to old single-vector embeddings when necessary.

    Returns a display score from 0 to 100.
    This score is NOT a probability of an exact match.
    """

    # --------------------------------------------------
    # 1. Read global embeddings
    # --------------------------------------------------
    stored_globals = prototype_document.get("global_embeddings")

    if stored_globals is not None and len(stored_globals) > 0:
        globals_array = np.asarray(stored_globals, dtype=np.float32)
    else:
        old_embedding = prototype_document.get(
            "global_embedding", prototype_document.get("embedding")
        )

        if old_embedding is None:
            return 0.0

        globals_array = np.asarray(old_embedding, dtype=np.float32).reshape(1, -1)

    query_globals = np.asarray(query_globals, dtype=np.float32)

    if query_globals.ndim == 1:
        query_globals = query_globals.reshape(1, -1)

    if query_globals.size == 0:
        return 0.0

    # --------------------------------------------------
    # 2. Global similarity across query views
    # --------------------------------------------------
    global_scores = []

    for query_global in query_globals:
        score = global_similarity(query_global, globals_array)

        if score >= -1.0:
            global_scores.append(score)

    if not global_scores:
        return 0.0

    global_score = max(global_scores)

    # --------------------------------------------------
    # 3. Local patch similarity across all views
    # --------------------------------------------------
    stored_patches = prototype_document.get("patch_embeddings")

    local_scores = []

    if stored_patches is not None:
        prototype_views = np.asarray(stored_patches, dtype=np.float32)

        query_patches = np.asarray(query_patches, dtype=np.float32)

        if prototype_views.ndim == 3 and query_patches.ndim == 3:
            for query_view_patches in query_patches:
                for prototype_view_patches in prototype_views:
                    score = local_patch_similarity(
                        query_view_patches, prototype_view_patches
                    )

                    if score >= -1.0:
                        local_scores.append(score)

    # --------------------------------------------------
    # 4. Combine scores
    # --------------------------------------------------
    if local_scores:
        local_score = max(local_scores)

        # Initial weights: validate and tune against
        # your verified matching pairs.
        combined_score = 0.60 * global_score + 0.40 * local_score
    else:
        # Fallback for older embedding documents.
        combined_score = global_score

    # Convert cosine similarity to a display scale.
    # This is not a calibrated probability.
    display_percent = (combined_score + 1.0) * 50.0

    return float(np.clip(display_percent, 0.0, 100.0))
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)