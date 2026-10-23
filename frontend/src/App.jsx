import { useEffect, useState } from "react";
import "./App.css";
const API = "/api";

export default function App() {
  const [file, setFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [matches, setMatches] = useState([]);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [loading, setLoading] = useState(false);
  const [checkingApi, setCheckingApi] = useState(true);
  const [apiStatus, setApiStatus] = useState("Checking...");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");
  const [dragging, setDragging] = useState(false);

  useEffect(() => {
    let active = true;

    async function checkApi() {
      try {
        const response = await fetch(`${API}/health`);

        if (!response.ok) {
          throw new Error("Matching API is not available.");
        }

        const data = await response.json();

        if (active) {
          setApiStatus(data.status === "ok" ? "Connected" : "Check API status");
        }
      } catch {
        if (active) {
          setApiStatus("Disconnected");
        }
      } finally {
        if (active) {
          setCheckingApi(false);
        }
      }
    }

    checkApi();

    return () => {
      active = false;
    };
  }, []);

  // Revoke temporary browser preview URLs when replaced or unmounted.
  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  function chooseImage(nextFile) {
    if (!nextFile) return;

    if (!nextFile.type.startsWith("image/")) {
      setError("Please choose a valid image file.");
      return;
    }

    if (nextFile.size > 15 * 1024 * 1024) {
      setError("Please choose an image smaller than 15 MB.");
      return;
    }

    setFile(nextFile);
    setPreviewUrl(URL.createObjectURL(nextFile));
    setMatches([]);
    setSelectedMatch(null);
    setError("");
    setMessage("");
  }

  function handleFileChange(event) {
    chooseImage(event.target.files?.[0]);
    // Allow selecting the same file again.
    event.target.value = "";
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseImage(event.dataTransfer.files?.[0]);
  }

  async function findMatches() {
    if (!file) {
      setError("Upload a gold jewellery image first.");
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");
    setMatches([]);
    setSelectedMatch(null);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(`${API}/match`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Unable to find matching designs.");
      }

      const results = Array.isArray(data.matches) ? data.matches : [];

      setMatches(results);

      if (results.length === 0) {
        setMessage("No valid prototype embeddings were found.");
      } else {
        setMessage(
          `Found ${results.length} matching prototype ${
            results.length === 1 ? "design" : "designs"
          }.`,
        );
      }
    } catch (err) {
      setError(
        err.message ||
          "Could not connect to the matching API. Check that Flask is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  function resetSearch() {
    setFile(null);
    setPreviewUrl("");
    setMatches([]);
    setSelectedMatch(null);
    setError("");
    setMessage("");
  }

  function formatScore(match) {
    const score = Number(match.similarityPercent);

    if (Number.isFinite(score)) {
      return `${score.toFixed(2)}%`;
    }

    const rawScore = Number(match.similarity);

    return Number.isFinite(rawScore)
      ? `${(rawScore * 100).toFixed(2)}%`
      : "N/A";
  }

  function closeOverlay(event) {
    if (event.target === event.currentTarget) {
      setSelectedMatch(null);
    }
  }

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape") {
        setSelectedMatch(null);
      }
    }

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  return (
    <main className="page">
      <header className="header">
        <div>
          <p className="eyebrow">INTERNAL DEVELOPMENT TOOL</p>
          <h1>Jewellery Model 3</h1>
          <p className="subtitle">
            Find the five closest prototype designs using DINOv2 embeddings
            stored in MongoDB.
          </p>
        </div>

        <div className="header-actions">
          <span
            className={`api-status ${
              apiStatus === "Connected" ? "online" : "offline"
            }`}
          >
            <span className="status-dot" />
            {checkingApi ? "Checking API..." : apiStatus}
          </span>

          <button
            className="secondary"
            type="button"
            onClick={resetSearch}
            disabled={loading}
          >
            New search
          </button>
        </div>
      </header>

      <section className="stats">
        <div className="stat">
          <span>Embedding model</span>
          <strong>DINOv2 Small</strong>
        </div>

        <div className="stat">
          <span>Embedding dimensions</span>
          <strong>384</strong>
        </div>

        <div className="stat">
          <span>Results displayed</span>
          <strong>{matches.length} / 5</strong>
        </div>
      </section>

      {error && (
        <div className="notice error" role="alert">
          {error}
        </div>
      )}

      {message && !error && (
        <div className="notice success" role="status">
          {message}
        </div>
      )}

      <section className="review-layout search-layout">
        <div className="gold-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">GOLD / QUERY IMAGE</p>
              <h2>Reference design</h2>
            </div>
          </div>

          <div
            className={`gold-image-wrap upload-dropzone ${
              dragging ? "dragging" : ""
            } ${previewUrl ? "has-image" : ""}`}
            onDragOver={(event) => {
              event.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={handleDrop}
          >
            {previewUrl ? (
              <img
                className="gold-image"
                src={previewUrl}
                alt="Uploaded gold jewellery reference"
              />
            ) : (
              <div className="upload-placeholder">
                <div className="upload-icon">↑</div>
                <h3>Upload a gold image</h3>
                <p>Drag and drop an image here or browse your computer.</p>
                <label className="secondary file-picker">
                  Choose image
                  <input
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                  />
                </label>
                <span className="muted small">
                  JPG, PNG or WEBP · Maximum 15 MB
                </span>
              </div>
            )}
          </div>

          {previewUrl && (
            <>
              <p className="filename" title={file?.name}>
                {file?.name}
              </p>

              <p className="muted small">
                {file ? `${(file.size / 1024 / 1024).toFixed(2)} MB` : ""}
              </p>

              <label className="secondary file-picker change-image">
                Change image
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                />
              </label>
            </>
          )}

          <button
            className="match-button find-matches-button"
            type="button"
            onClick={findMatches}
            disabled={!file || loading}
          >
            {loading ? "Finding matches..." : "Find Top 5 Matches"}
          </button>

          {loading && (
            <p className="muted loading-message">
              Processing image and comparing saved MongoDB embeddings...
            </p>
          )}

          <p className="muted small privacy-note">
            The query image is processed in memory. This search does not save it
            to MongoDB.
          </p>
        </div>

        <div className="prototype-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">DINOv2 SEARCH RESULTS</p>
              <h2>Closest prototypes</h2>
            </div>

            <span className="counter">{matches.length} matches</span>
          </div>

          {!loading && matches.length === 0 && (
            <div className="empty-results">
              <div className="empty-results-icon">⌕</div>
              <h3>Your matches will appear here</h3>
              <p>
                Upload a gold jewellery image, then select Find Top 5 Matches.
              </p>
            </div>
          )}

          {loading && (
            <div className="empty-results">
              <div className="loading-spinner" />
              <h3>Searching the prototype catalogue</h3>
              <p>Comparing the query with the saved DINOv2 vectors.</p>
            </div>
          )}

          {!loading && matches.length > 0 && (
            <div className="prototype-grid result-grid">
              {matches.slice(0, 5).map((item, index) => (
                <button
                  type="button"
                  key={`${item.catalogueId || item.filename}-${index}`}
                  className="prototype-card result-card"
                  onClick={() => setSelectedMatch(item)}
                >
                  <div className="prototype-image-wrap">
                    {item.imageUrl ? (
                      <img
                        src={item.imageUrl}
                        alt={item.name || item.catalogueId || "Prototype"}
                        loading="lazy"
                      />
                    ) : (
                      <div className="image-unavailable">Image unavailable</div>
                    )}

                    <span className="result-rank">#{index + 1}</span>
                  </div>

                  <span className="prototype-name">
                    {item.name || "Unnamed design"}
                  </span>

                  <span className="muted small">
                    ID: {item.catalogueId || "N/A"}
                  </span>

                  {item.type && (
                    <span className="muted small">Type: {item.type}</span>
                  )}

                  <div className="similarity-row">
                    <span>Similarity</span>
                    <strong>{formatScore(item)}</strong>
                  </div>

                  <div
                    className="similarity-track"
                    aria-label={`Similarity ${formatScore(item)}`}
                  >
                    <span
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(
                            100,
                            Number(item.similarityPercent) ||
                              Number(item.similarity) * 100 ||
                              0,
                          ),
                        )}%`,
                      }}
                    />
                  </div>

                  <span className="view-result">View details ↗</span>
                </button>
              ))}
            </div>
          )}
        </div>
      </section>

      <footer className="page-footer">
        <span>JEWELLERY MODEL 3</span>
        <span>
          MongoDB GridFS prototypes · DINOv2 Small · Cosine similarity
        </span>
      </footer>

      {selectedMatch && (
        <div
          className="result-overlay"
          role="presentation"
          onClick={closeOverlay}
        >
          <section
            className="result-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Prototype match details"
          >
            <button
              type="button"
              className="overlay-close"
              onClick={() => setSelectedMatch(null)}
              aria-label="Close details"
            >
              ×
            </button>

            <div className="overlay-image-wrap">
              {selectedMatch.imageUrl && (
                <img
                  src={selectedMatch.imageUrl}
                  alt={selectedMatch.name || "Prototype match"}
                />
              )}
            </div>

            <div className="overlay-details">
              <p className="eyebrow">PROTOTYPE MATCH DETAILS</p>
              <h2>{selectedMatch.name || "Unnamed design"}</h2>
              <p className="muted">
                Catalogue ID: {selectedMatch.catalogueId || "N/A"}
              </p>

              <div className="detail-row">
                <span>Similarity score</span>
                <strong>{formatScore(selectedMatch)}</strong>
              </div>

              <div className="detail-row">
                <span>Cosine similarity</span>
                <strong>
                  {Number.isFinite(Number(selectedMatch.similarity))
                    ? Number(selectedMatch.similarity).toFixed(4)
                    : "N/A"}
                </strong>
              </div>

              <div className="detail-row">
                <span>Jewellery type</span>
                <strong>{selectedMatch.type || "N/A"}</strong>
              </div>

              <div className="detail-row">
                <span>Image filename</span>
                <strong>{selectedMatch.filename || "N/A"}</strong>
              </div>

              <button
                type="button"
                className="secondary close-details"
                onClick={() => setSelectedMatch(null)}
              >
                Close details
              </button>
            </div>
          </section>
        </div>
      )}
    </main>
  );
}
