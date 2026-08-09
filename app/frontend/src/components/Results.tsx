import { useState } from "react";
import type { CombinedResult, Mode } from "../types";
import Gauge from "./Gauge";

interface Props {
  result: CombinedResult;
  mode: Mode;
  mriURL: string | null;
  onNewScreening: () => void;
}

export default function Results({
  result: r,
  mode,
  mriURL,
  onNewScreening,
}: Props) {
  const [heatmapOn, setHeatmapOn] = useState(true);

  const modalityLabel =
    mode === "voice" ? "Voice" : mode === "mri" ? "MRI" : "Combined";

  const expl = r.explanation;
  const method = expl.method || "SHAP";
  const features = expl.features ?? [];
  const gradcam = r.gradcam ?? "";

  const showFeatures =
    (mode === "voice" || mode === "combined") && features.length > 0;
  const showHeatmap = mode === "mri" || mode === "combined";
  const cols = (showFeatures ? 1 : 0) + (showHeatmap ? 1 : 0) + 1;
  const explGrid =
    cols === 3
      ? "minmax(0,1fr) minmax(0,1fr) minmax(0,1fr)"
      : cols === 2
        ? "minmax(0,1fr) minmax(0,1fr)"
        : "minmax(0,1fr)";

  const caveatCol =
    mode === "mri"
      ? { bg: "#F2EEF8", border: "#E0D6F0", fg: "#5B3E86" }
      : { bg: "#F2EEE2", border: "#E4DAC2", fg: "#7A5E22" };

  const rec = r.recommendation;
  const hasSecondary = !!rec.secondary;

  return (
    <section
      aria-labelledby="nv-result-h"
      className="nv-screen"
      style={{ opacity: 1 }}
    >
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 14,
          marginBottom: 20,
        }}
      >
        <div>
          <div
            style={{
              fontFamily: "'IBM Plex Mono',monospace",
              fontSize: 11.5,
              letterSpacing: ".1em",
              textTransform: "uppercase",
              color: "#7A8784",
              marginBottom: 6,
            }}
          >
            {modalityLabel} screening result
          </div>
          <h1
            id="nv-result-h"
            style={{
              fontSize: 28,
              fontWeight: 700,
              letterSpacing: "-.01em",
              margin: 0,
            }}
          >
            Risk estimate
          </h1>
        </div>
        <div style={{ display: "flex", gap: 10 }}>
          <button
            data-nv-focusring
            onClick={onNewScreening}
            style={{
              background: "#fff",
              border: "1.5px solid #CADEDB",
              borderRadius: 10,
              height: 44,
              padding: "0 16px",
              font: "600 14px 'IBM Plex Sans'",
              color: "#0C5C5E",
              cursor: "pointer",
            }}
          >
            New screening
          </button>
        </div>
      </div>

      {r.caveat && (
        <div
          role="note"
          style={{
            display: "flex",
            gap: 11,
            alignItems: "flex-start",
            background: caveatCol.bg,
            border: `1px solid ${caveatCol.border}`,
            borderRadius: 12,
            padding: "14px 16px",
            marginBottom: 20,
          }}
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            aria-hidden="true"
            style={{ flex: "none", marginTop: 1 }}
          >
            <path
              d="M12 3 22 20H2L12 3Z"
              stroke={caveatCol.fg}
              strokeWidth="2"
              strokeLinejoin="round"
            />
            <path d="M12 10v4" stroke={caveatCol.fg} strokeWidth="2" strokeLinecap="round" />
            <circle cx="12" cy="17" r="1.1" fill={caveatCol.fg} />
          </svg>
          <p style={{ margin: 0, fontSize: 14, lineHeight: 1.5, color: caveatCol.fg }}>
            {r.caveat}
          </p>
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0,1fr) minmax(0,1fr)",
          gap: 18,
          alignItems: "start",
        }}
      >
        {/* GAUGE CARD */}
        <Gauge probability={r.probability} band={r.risk_band} />

        {/* CARE ROUTE CARD */}
        <div
          style={{
            background: "#0C5C5E",
            color: "#EAF4F2",
            borderRadius: 16,
            boxShadow: "0 1px 2px rgba(20,40,40,.04)",
            padding: 24,
            display: "flex",
            flexDirection: "column",
          }}
        >
          <div
            style={{
              fontFamily: "'IBM Plex Mono',monospace",
              fontSize: 11.5,
              letterSpacing: ".1em",
              textTransform: "uppercase",
              color: "#9CC9C5",
              marginBottom: 14,
            }}
          >
            Suggested care route
          </div>
          <h2
            style={{
              fontSize: 23,
              fontWeight: 700,
              lineHeight: 1.2,
              margin: "0 0 16px",
              color: "#fff",
            }}
          >
            {rec.route}
          </h2>
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 10,
              marginBottom: 18,
            }}
          >
            <RouteRow
              label="Refer to"
              value={rec.specialist_type}
              icon={
                <>
                  <circle cx="12" cy="8" r="3.6" stroke="#fff" strokeWidth="2" />
                  <path
                    d="M5 20c0-3.6 3.1-6 7-6s7 2.4 7 6"
                    stroke="#fff"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </>
              }
            />
            {hasSecondary && (
              <RouteRow
                label="Also consider"
                value={rec.secondary}
                icon={
                  <path
                    d="M12 21s-7-4.5-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 11c0 5.5-7 10-7 10Z"
                    stroke="#fff"
                    strokeWidth="2"
                    strokeLinejoin="round"
                  />
                }
              />
            )}
          </div>
          <p
            style={{
              fontSize: 13,
              lineHeight: 1.55,
              color: "#BFDCD9",
              margin: "auto 0 0",
              paddingTop: 14,
              borderTop: "1px solid rgba(255,255,255,.14)",
            }}
          >
            {rec.disclaimer}
          </p>
        </div>

        {/* EXPLANATION (spans both columns) */}
        <div
          style={{
            gridColumn: "1 / -1",
            background: "#fff",
            border: "1px solid #EAE3D5",
            borderRadius: 16,
            boxShadow: "0 1px 2px rgba(20,40,40,.04)",
            padding: 24,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 9,
              marginBottom: 6,
            }}
          >
            <h2 style={{ fontSize: 15, fontWeight: 700, margin: 0 }}>
              Why this estimate
            </h2>
            <span
              style={{
                fontFamily: "'IBM Plex Mono',monospace",
                fontSize: 10.5,
                fontWeight: 500,
                color: "#0C5C5E",
                background: "#E6F0EE",
                padding: "3px 8px",
                borderRadius: 6,
              }}
            >
              {method}
            </span>
          </div>
          <p style={{ fontSize: 13, color: "#7A8784", margin: "0 0 18px" }}>
            A plain-language summary, with the model&rsquo;s main signals.
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: explGrid,
              gap: 24,
              alignItems: "start",
            }}
          >
            {/* SHAP features */}
            {showFeatures && (
              <div>
                <h3
                  style={{
                    fontSize: 13,
                    fontWeight: 600,
                    color: "#5A6A68",
                    margin: "0 0 14px",
                    textTransform: "uppercase",
                    letterSpacing: ".04em",
                  }}
                >
                  Top contributing features
                </h3>
                <div style={{ display: "flex", flexDirection: "column", gap: 13 }}>
                  {features.map((f, i) => {
                    const barColor = f.dir === "down" ? "#0A6E5C" : "#9A5B00";
                    const dirText = f.dir === "down" ? "↓ lowers" : "↑ raises";
                    return (
                      <div key={i}>
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            gap: 10,
                            marginBottom: 5,
                          }}
                        >
                          <span style={{ fontSize: 13.5, fontWeight: 500 }}>
                            {f.name}
                          </span>
                          <span
                            aria-hidden="true"
                            style={{
                              fontSize: 11,
                              fontWeight: 600,
                              color: barColor,
                              whiteSpace: "nowrap",
                            }}
                          >
                            {dirText}
                          </span>
                        </div>
                        <div
                          style={{
                            height: 10,
                            borderRadius: 5,
                            background: "#EFE9DD",
                            overflow: "hidden",
                          }}
                        >
                          <span
                            style={{
                              display: "block",
                              height: "100%",
                              borderRadius: 5,
                              background: barColor,
                              width: `${Math.round(f.weight * 100)}%`,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <p style={{ fontSize: 11.5, color: "#9AA6A3", margin: "14px 0 0" }}>
                  Bars show relative SHAP contribution.{" "}
                  <span style={{ color: "#0A6E5C", fontWeight: 600 }}>↓ lowers</span> and{" "}
                  <span style={{ color: "#9A5B00", fontWeight: 600 }}>↑ raises</span>{" "}
                  estimated risk.
                </p>
              </div>
            )}

            {/* Grad-CAM */}
            {showHeatmap && (
              <div>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 14,
                  }}
                >
                  <h3
                    style={{
                      fontSize: 13,
                      fontWeight: 600,
                      color: "#5A6A68",
                      margin: 0,
                      textTransform: "uppercase",
                      letterSpacing: ".04em",
                    }}
                  >
                    Grad-CAM overlay
                  </h3>
                  <button
                    data-nv-focusring
                    onClick={() => setHeatmapOn((v) => !v)}
                    aria-pressed={heatmapOn}
                    style={{
                      background: "#fff",
                      border: "1.5px solid #CADEDB",
                      borderRadius: 8,
                      height: 32,
                      padding: "0 11px",
                      font: "600 12px 'IBM Plex Sans'",
                      color: "#0C5C5E",
                      cursor: "pointer",
                    }}
                  >
                    {heatmapOn ? "Hide heatmap" : "Show heatmap"}
                  </button>
                </div>
                <div
                  style={{
                    position: "relative",
                    borderRadius: 12,
                    overflow: "hidden",
                    border: "1px solid #E4DDCE",
                    background: "#0a0f12",
                    aspectRatio: "1 / 1",
                    maxWidth: 320,
                  }}
                >
                  {gradcam && heatmapOn ? (
                    <img
                      src={gradcam}
                      alt="Grad-CAM overlay on the MRI"
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        background: "#0a0f12",
                      }}
                    />
                  ) : mriURL ? (
                    <img
                      src={mriURL}
                      alt="MRI under analysis"
                      style={{
                        position: "absolute",
                        inset: 0,
                        width: "100%",
                        height: "100%",
                        objectFit: "cover",
                        filter: "grayscale(1) contrast(1.05)",
                      }}
                    />
                  ) : (
                    <div
                      aria-hidden="true"
                      style={{
                        position: "absolute",
                        inset: 0,
                        background:
                          "repeating-linear-gradient(45deg,#1b2329,#1b2329 9px,#222c33 9px,#222c33 18px)",
                        display: "grid",
                        placeItems: "center",
                      }}
                    >
                      <span
                        style={{
                          fontFamily: "'IBM Plex Mono',monospace",
                          fontSize: 11,
                          color: "#5b6b72",
                        }}
                      >
                        brain MRI · axial
                      </span>
                    </div>
                  )}
                  {heatmapOn && !gradcam && (
                    <div
                      aria-hidden="true"
                      style={{
                        position: "absolute",
                        inset: 0,
                        background:
                          "radial-gradient(circle at 54% 58%,rgba(214,80,28,.85),rgba(214,80,28,.45) 9%,rgba(230,160,0,.4) 16%,rgba(0,158,115,.25) 26%,transparent 38%),radial-gradient(circle at 44% 56%,rgba(214,80,28,.6),transparent 14%)",
                        mixBlendMode: "screen",
                      }}
                    />
                  )}
                  <div
                    style={{
                      position: "absolute",
                      left: 10,
                      bottom: 10,
                      right: 10,
                      display: "flex",
                      alignItems: "center",
                      gap: 8,
                    }}
                  >
                    <span
                      style={{
                        fontFamily: "'IBM Plex Mono',monospace",
                        fontSize: 9.5,
                        color: "#cfd8dc",
                      }}
                    >
                      low
                    </span>
                    <span
                      aria-hidden="true"
                      style={{
                        flex: 1,
                        height: 7,
                        borderRadius: 4,
                        background:
                          "linear-gradient(90deg,#0a4f3f,#009E73,#E6A000,#D6501C)",
                      }}
                    />
                    <span
                      style={{
                        fontFamily: "'IBM Plex Mono',monospace",
                        fontSize: 9.5,
                        color: "#cfd8dc",
                      }}
                    >
                      high activation
                    </span>
                  </div>
                </div>
                <p style={{ fontSize: 11.5, color: "#9AA6A3", margin: "12px 0 0" }}>
                  Grad-CAM shows where the model looked, not a radiological read. Because this
                  baseline exploits scanner and image artefacts, its attention often falls
                  outside the brain, which is the leakage finding it is here to illustrate.
                </p>
              </div>
            )}

            {/* Plain language */}
            <div>
              <h3
                style={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: "#5A6A68",
                  margin: "0 0 12px",
                  textTransform: "uppercase",
                  letterSpacing: ".04em",
                }}
              >
                In plain language
              </h3>
              <div
                style={{
                  background: "#F4F8F7",
                  border: "1px solid #DCEAE7",
                  borderRadius: 12,
                  padding: 18,
                }}
              >
                <p
                  style={{
                    fontSize: 15.5,
                    lineHeight: 1.6,
                    margin: 0,
                    color: "#234240",
                  }}
                >
                  {expl.plain_text}
                </p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <p
        style={{
          textAlign: "center",
          fontSize: 12.5,
          color: "#9AA6A3",
          margin: "26px auto 0",
          maxWidth: "60ch",
        }}
      >
        This estimate supports — and does not replace — clinical judgement and a full
        assessment. LUCID-PD CDSS is not a medical device and must not be the sole basis
        for any care decision.
      </p>
    </section>
  );
}

function RouteRow({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 11,
        background: "rgba(255,255,255,.08)",
        borderRadius: 11,
        padding: "12px 14px",
      }}
    >
      <span
        aria-hidden="true"
        style={{
          width: 34,
          height: 34,
          borderRadius: 9,
          background: "rgba(255,255,255,.14)",
          display: "grid",
          placeItems: "center",
          flex: "none",
        }}
      >
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
          {icon}
        </svg>
      </span>
      <div>
        <div
          style={{
            fontSize: 11,
            color: "#9CC9C5",
            textTransform: "uppercase",
            letterSpacing: ".05em",
          }}
        >
          {label}
        </div>
        <div style={{ fontSize: 15, fontWeight: 600, color: "#fff" }}>{value}</div>
      </div>
    </div>
  );
}
