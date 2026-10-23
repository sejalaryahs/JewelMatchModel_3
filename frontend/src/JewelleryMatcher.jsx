import { useRef, useState } from "react";
import "./JewelleryMatcher.css";

const API_URL = "http://127.0.0.1:5000";

export default function JewelleryMatcher() {
  const fileInputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [matches, setMatches] = useState([]);
  const [selectedMatch, setSelectedMatch] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [dragging, setDragging] = useState(false);

  function chooseFile(nextFile) {
    if (!nextFile) return;

    if (!nextFile.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (nextFile.size > 15 * 1024 * 1024) {
      setError("The image must be smaller than 15 MB.");
      return;
    }

    setError("");
    setMatches([]);
    setFile(nextFile);
    setPreview(URL.createObjectURL(nextFile));
  }

  function handleFileChange(event) {
    chooseFile(event.target.files?.[0]);
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  }

  async function findMatches() {
    if (!file) {
      setError("Upload a gold jewellery image first.");
      return;
    }

    setLoading(true);
    setError("");
    setMatches([]);

    try {
      const formData = new FormData();
      formData.append("image", file);

      const response = await fetch(`${API_URL}/api/match`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Matching failed.");
      }

      setMatches(data.matches || []);

      if (!data.matches?.length) {
        setError("No prototype matches were returned.");
      }
    } catch (err) {
      setError(
        err.message ||
          "Cannot connect to the matching API. Check that Flask is running.",
      );
    } finally {
      setLoading(false);
    }
  }

  function resetSearch() {
    setFile(null);
    setPreview("");
    setMatches([]);
    setError("");
    if (fileInputRef.current) fileInputRef.current.value = "";
  }

  return (
    <div className="jm-app">
      <header className="jm-header">
        <a className="jm-brand" href="/" aria-label="Jewellery Model 3 home">
          <span className="jm-brand-mark">J</span>
          <span>
            <strong>JEWELMATCH</strong>
            <small>MODEL 3 · VISUAL SEARCH</small>
          </span>
        </a>
        <div className="jm-status">
          <span className="jm-status-dot" />
          DINOv2 Matching
        </div>
      </header>

      <main className="jm-main">
        <section className="jm-intro">
          <p className="jm-eyebrow">INTELLIGENT JEWELLERY DISCOVERY</p>
          <h1>
            Find the closest
            <br />
            <em>design match.</em>
          </h1>
          <p className="jm-subtitle">
            Upload a gold jewellery image and discover visually similar designs
            from your prototype catalogue.
          </p>
        </section>

        <section className="jm-workspace">
          <div className="jm-upload-panel">
            <div className="jm-section-heading">
              <span className="jm-step">01</span>
              <div>
                <h2>Upload reference image</h2>
                <p>Choose a clear image of the jewellery design.</p>
              </div>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              hidden
              onChange={handleFileChange}
            />

            <div
              className={`jm-dropzone ${dragging ? "is-dragging" : ""} ${
                preview ? "has-preview" : ""
              }`}
              onDragOver={(event) => {
                event.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
            >
              {preview ? (
                <div className="jm-preview-wrap">
                  <img
                    src={preview}
                    alt="Uploaded gold jewellery"
                    className="jm-query-preview"
                  />
                  <button
                    type="button"
                    className="jm-change-image"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Change image
                  </button>
                </div>
              ) : (
                <>
                  <div className="jm-upload-icon">↑</div>
                  <h3>Drop your image here</h3>
                  <p>or browse your computer</p>
                  <button
                    type="button"
                    className="jm-secondary-button"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    Choose image
                  </button>
                  <span className="jm-file-note">
                    JPG, PNG, WEBP · Max 15 MB
                  </span>
                </>
              )}
            </div>

            {file && (
              <div className="jm-file-info">
                <span className="jm-file-symbol">▧</span>
                <div className="jm-file-text">
                  <strong>{file.name}</strong>
                  <small>{(file.size / 1024 / 1024).toFixed(2)} MB</small>
                </div>
                <button
                  type="button"
                  className="jm-remove-file"
                  onClick={resetSearch}
                  aria-label="Remove image"
                >
                  ×
                </button>
              </div>
            )}

            <button
              type="button"
              className="jm-primary-button"
              onClick={findMatches}
              disabled={!file || loading}
            >
              {loading ? (
                <>
                  <span className="jm-spinner" />
                  Finding matches...
                </>
              ) : (
                <>
                  Find matching designs <span>→</span>
                </>
              )}
            </button>

            <p className="jm-privacy-note">
              Your query image is processed in memory and is not stored in
              MongoDB by this matching API.
            </p>
          </div>

          <aside className="jm-info-panel">
            <div className="jm-info-art" aria-hidden="true">
              <span className="jm-art-ring jm-ring-one" />
              <span className="jm-art-ring jm-ring-two" />
              <span className="jm-art-spark">✦</span>
            </div>
            <p className="jm-eyebrow">HOW IT WORKS</p>
            <h2>From image to insight.</h2>
            <div className="jm-process-list">
              <div>
                <span>01</span>
                <p>
                  <strong>Upload</strong>
                  <small>Select your gold design.</small>
                </p>
              </div>
              <div>
                <span>02</span>
                <p>
                  <strong>Analyse</strong>
                  <small>DINOv2 extracts visual features.</small>
                </p>
              </div>
              <div>
                <span>03</span>
                <p>
                  <strong>Discover</strong>
                  <small>Rank the five closest prototypes.</small>
                </p>
              </div>
            </div>
            <div className="jm-model-tag">
              <span className="jm-status-dot" />
              Powered by DINOv2 Small
            </div>
          </aside>
        </section>

        {error && (
          <div className="jm-error" role="alert">
            <strong>Something needs attention</strong>
            <span>{error}</span>
          </div>
        )}

        {(matches.length > 0 || loading) && (
          <section className="jm-results">
            <div className="jm-results-heading">
              <div>
                <p className="jm-eyebrow">YOUR CATALOGUE RESULTS</p>
                <h2>Closest design matches</h2>
                <p>Ranked by DINOv2 cosine similarity.</p>
              </div>
              {matches.length > 0 && (
                <span className="jm-result-count">
                  {matches.length} MATCHES
                </span>
              )}
            </div>

            {loading ? (
              <div className="jm-loading-state">
                <span className="jm-spinner jm-spinner-large" />
                <strong>Comparing jewellery designs...</strong>
                <p>Checking your image against the prototype catalogue.</p>
              </div>
            ) : (
              <div className="jm-results-grid">
                {matches.map((match, index) => (
                  <button
                    type="button"
                    className="jm-result-card"
                    key={match.catalogueId || index}
                    onClick={() => setSelectedMatch(match)}
                  >
                    <div className="jm-result-image">
                      {match.imageUrl ? (
                        <img
                          src={match.imageUrl}
                          alt={match.name || match.catalogueId}
                          loading="lazy"
                        />
                      ) : (
                        <div className="jm-no-image">Image unavailable</div>
                      )}
                      <span className="jm-rank">#{index + 1}</span>
                    </div>
                    <div className="jm-result-body">
                      <div className="jm-result-id">
                        {match.catalogueId || "No catalogue ID"}
                      </div>
                      <h3>{match.name || "Unnamed design"}</h3>
                      <p className="jm-result-type">
                        {match.type || "Prototype"}
                      </p>
                      <div className="jm-score-row">
                        <span>Similarity score</span>
                        <strong>
                          {Number(match.similarityPercent).toFixed(2)}%
                        </strong>
                      </div>
                      <div className="jm-score-track">
                        <span
                          style={{
                            width: `${Math.max(
                              0,
                              Math.min(
                                100,
                                Number(match.similarityPercent) || 0,
                              ),
                            )}%`,
                          }}
                        />
                      </div>
                      <span className="jm-view-details">View details ↗</span>
                    </div>
                  </button>
                ))}
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="jm-footer">
        <span>JEWELMATCH · MODEL 3</span>
        <span>
          Visual similarity is a ranking aid, not a guarantee of identical
          design.
        </span>
      </footer>

      {selectedMatch && (
        <div
          className="jm-modal-backdrop"
          role="presentation"
          onClick={() => setSelectedMatch(null)}
        >
          <div
            className="jm-modal"
            role="dialog"
            aria-modal="true"
            aria-label="Prototype details"
            onClick={(event) => event.stopPropagation()}
          >
            <button
              type="button"
              className="jm-modal-close"
              onClick={() => setSelectedMatch(null)}
              aria-label="Close details"
            >
              ×
            </button>
            {selectedMatch.imageUrl && (
              <img
                className="jm-modal-image"
                src={selectedMatch.imageUrl}
                alt={selectedMatch.name}
              />
            )}
            <div className="jm-modal-content">
              <p className="jm-eyebrow">PROTOTYPE DETAILS</p>
              <h2>{selectedMatch.name || "Unnamed design"}</h2>
              <p className="jm-modal-id">{selectedMatch.catalogueId}</p>
              <div className="jm-modal-score">
                <span>Cosine similarity</span>
                <strong>{Number(selectedMatch.similarity).toFixed(4)}</strong>
              </div>
              <div className="jm-modal-score">
                <span>Displayed score</span>
                <strong>
                  {Number(selectedMatch.similarityPercent).toFixed(2)}%
                </strong>
              </div>
              <div className="jm-modal-score">
                <span>Type</span>
                <strong>{selectedMatch.type || "—"}</strong>
              </div>
              <div className="jm-modal-score">
                <span>Image file</span>
                <strong>{selectedMatch.filename || "—"}</strong>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
