import { useEffect, useMemo, useState } from "react";
import "./BaselineDashboard.css";

const API = "/dashboard-api";

function scoreText(score) {
  return Number(score).toFixed(4);
}

export default function BaselineDashboard() {
  const [items, setItems] = useState([]);
  const [selected, setSelected] = useState(0);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    fetch(`${API}/api/dashboard`)
      .then(async (response) => {
        const data = await response.json();
        if (!response.ok)
          throw new Error(data.error || "Could not load results.");
        return data;
      })
      .then((data) => {
        setItems(data.results || []);
        setError("");
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    const query = search.trim().toLowerCase();
    if (!query) return items;
    return items.filter((item) =>
      item.gold_filename.toLowerCase().includes(query),
    );
  }, [items, search]);

  const current = filtered[selected] || null;

  useEffect(() => {
    setSelected(0);
  }, [search]);

  const verifiedCount = current
    ? current.matches.filter((match) => match.verified_label !== null).length
    : 0;

  if (loading) {
    return (
      <main className="dashboard">
        <div className="status">Loading baseline results…</div>
      </main>
    );
  }

  if (error) {
    return (
      <main className="dashboard">
        <div className="error-panel">
          <h2>Could not load the dashboard</h2>
          <p>{error}</p>
          <p>Check that the Flask dashboard backend is running on port 5002.</p>
          <button onClick={() => window.location.reload()}>Try again</button>
        </div>
      </main>
    );
  }

  if (!items.length) {
    return (
      <main className="dashboard">
        <div className="error-panel">
          <h2>No ranking results found</h2>
          <p>
            Run the DINOv2 evaluation script first to generate
            dinov2_top5_results.csv.
          </p>
        </div>
      </main>
    );
  }

  return (
    <main className="dashboard">
      <header className="dashboard-header">
        <div>
          <div className="eyebrow">JEWELLERY MODEL 3</div>
          <h1>Visual matching dashboard</h1>
          <p>
            Explore what the pretrained DINOv2 baseline retrieves for each gold
            image.
          </p>
        </div>
        <div className="baseline-pill">
          <span /> Pretrained baseline
        </div>
      </header>

      <section className="summary-grid">
        <div className="summary-card">
          <span>Gold queries</span>
          <strong>{items.length}</strong>
          <small>Images being searched</small>
        </div>
        <div className="summary-card">
          <span>Prototypes per query</span>
          <strong>5</strong>
          <small>Highest similarity scores</small>
        </div>
        <div className="summary-card">
          <span>Verified pairs shown</span>
          <strong>
            {verifiedCount} / {current?.matches.length || 0}
          </strong>
          <small>Labels already in your CSV</small>
        </div>
        <div className="summary-card">
          <span>Embedding model</span>
          <strong>DINOv2</strong>
          <small>Small · 384 dimensions</small>
        </div>
      </section>

      <section className="browser-toolbar">
        <div>
          <label htmlFor="gold-search">Find a gold image</label>
          <input
            id="gold-search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            placeholder="Search by filename…"
          />
        </div>
        <div className="query-navigation">
          <button
            disabled={selected <= 0}
            onClick={() => setSelected((value) => value - 1)}
          >
            ← Previous
          </button>
          <span>
            {filtered.length ? selected + 1 : 0} of {filtered.length}
          </span>
          <button
            disabled={selected >= filtered.length - 1}
            onClick={() => setSelected((value) => value + 1)}
          >
            Next →
          </button>
        </div>
      </section>

      {!current ? (
        <div className="status">No gold images match that search.</div>
      ) : (
        <>
          <section className="query-panel">
            <div className="query-image-wrap">
              <img
                src={`${API}${current.gold_image_url}`}
                alt={`Gold query ${current.gold_filename}`}
              />
            </div>
            <div className="query-info">
              <div className="eyebrow">QUERY IMAGE</div>
              <h2>{current.gold_filename}</h2>
              <p>
                The model compares this image with the prototype catalogue and
                ranks candidates by cosine similarity.
              </p>
              <div className="query-stats">
                <div>
                  <strong>214</strong>
                  <span>Catalogue prototypes</span>
                </div>
                <div>
                  <strong>384</strong>
                  <span>Features per image</span>
                </div>
                <div>
                  <strong>Top 5</strong>
                  <span>Results displayed</span>
                </div>
              </div>
            </div>
          </section>

          <div className="results-heading">
            <div>
              <div className="eyebrow">RANKED CANDIDATES</div>
              <h2>Closest prototype images</h2>
            </div>
            <span className="score-note">
              Higher cosine similarity = closer embedding
            </span>
          </div>

          <section className="matches-grid">
            {current.matches.map((match) => (
              <article
                className="match-card"
                key={`${match.rank}-${match.prototype_filename}`}
              >
                <div className="match-card-top">
                  <span className="rank-badge">#{match.rank}</span>
                  {match.verified_label === 1 && (
                    <span className="label-badge positive">Verified match</span>
                  )}
                  {match.verified_label === 0 && (
                    <span className="label-badge negative">
                      Verified non-match
                    </span>
                  )}
                  {match.verified_label === null && (
                    <span className="label-badge unknown">Not labelled</span>
                  )}
                </div>

                <div className="prototype-image-wrap">
                  <img
                    src={`${API}${match.prototype_image_url}`}
                    alt={`Rank ${match.rank}: ${match.prototype_filename}`}
                    loading="lazy"
                  />
                </div>

                <div className="match-details">
                  <span className="match-rank-text">RANK {match.rank}</span>
                  <h3>{match.prototype_filename}</h3>
                  <div className="similarity-row">
                    <span>Cosine similarity</span>
                    <strong>{scoreText(match.score)}</strong>
                  </div>
                  <div className="score-track">
                    <div
                      className="score-fill"
                      style={{
                        width: `${Math.max(0, Math.min(100, match.score * 100))}%`,
                      }}
                    />
                  </div>
                  {match.match_type && (
                    <p className="verified-note">Label: {match.match_type}</p>
                  )}
                </div>
              </article>
            ))}
          </section>

          <section className="how-to-read">
            <h3>How to read these results</h3>
            <ul>
              <li>
                <strong>Rank #1</strong> is the highest-scoring candidate for
                this query, not a guarantee of an exact design match.
              </li>
              <li>
                <strong>Cosine similarity</strong> measures the closeness of the
                image embeddings. It is not a probability or percentage chance
                of a match.
              </li>
              <li>
                <strong>Verified match</strong> and{" "}
                <strong>Verified non-match</strong> appear only when a
                corresponding label exists in your CSV.
              </li>
              <li>
                <strong>Not labelled</strong> means the pair has not been
                verified yet; it does not mean the pair is wrong.
              </li>
            </ul>
          </section>
        </>
      )}
    </main>
  );
}
