interface Props {
  onHome: () => void;
}

const card = (dot: string, title: string, body: React.ReactNode) => (
  <div
    style={{
      background: "#fff",
      border: "1px solid #EAE3D5",
      borderRadius: 14,
      padding: 22,
    }}
  >
    <h2
      style={{
        fontSize: 16,
        fontWeight: 700,
        margin: "0 0 8px",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <span style={{ width: 8, height: 8, borderRadius: "50%", background: dot }} />
      {title}
    </h2>
    <p style={{ fontSize: 14.5, lineHeight: 1.6, color: "#4A5A58", margin: 0 }}>
      {body}
    </p>
  </div>
);

export default function About({ onHome }: Props) {
  return (
    <section
      aria-labelledby="nv-about-h"
      className="nv-screen"
      style={{ maxWidth: 760 }}
    >
      <button
        data-nv-focusring
        onClick={onHome}
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 7,
          background: "none",
          border: "none",
          cursor: "pointer",
          font: "500 14px 'IBM Plex Sans'",
          color: "#5A6A68",
          padding: "6px 4px",
          marginBottom: 8,
        }}
      >
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M15 6l-6 6 6 6"
            stroke="#5A6A68"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        Back
      </button>
      <h1
        id="nv-about-h"
        style={{
          fontSize: 30,
          fontWeight: 700,
          letterSpacing: "-.01em",
          margin: "0 0 8px",
        }}
      >
        About &amp; methodology
      </h1>
      <p
        style={{
          fontSize: 17,
          lineHeight: 1.6,
          color: "#3C4D4B",
          margin: "0 0 28px",
        }}
      >
        LUCID-PD CDSS estimates the likelihood of speech and imaging patterns associated
        with Parkinson&rsquo;s disease. It is a{" "}
        <strong>decision-support and research tool</strong> for clinicians — not a
        diagnosis, and not a medical device.
      </p>

      <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
        {card(
          "#0A6E5C",
          "Voice — primary mode",
          "openSMILE converts a sustained vowel into the 88 eGeMAPS acoustic parameters, covering pitch, loudness, speech timing, articulation and spectral shape, and voice quality measures such as jitter, shimmer and harmonics-to-noise. A gradient boosted classifier scores them, and SHAP groups its attribution into those families so the reasoning is inspectable.",
        )}
        {card(
          "#6A4F8C",
          "MRI — research baseline",
          <>
            A convolutional model produces a Grad-CAM heatmap over the scan. This pathway
            is <strong>exploratory</strong>: it is not validated for clinical use and must
            not guide care on its own.
          </>,
        )}
        {card(
          "#8A6A2A",
          "Combined — illustrative only",
          "Voice and MRI inputs come from different sources and are not jointly modelled, so combined output is illustrative and should be read as such.",
        )}

        <div
          style={{
            background: "#F4F8F7",
            border: "1px solid #DCEAE7",
            borderRadius: 14,
            padding: 22,
          }}
        >
          <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 10px" }}>
            Ethics &amp; safety
          </h2>
          <ul
            style={{
              margin: 0,
              paddingLeft: 18,
              fontSize: 14.5,
              lineHeight: 1.7,
              color: "#234240",
            }}
          >
            <li>A clinician stays in the loop — output informs, never decides.</li>
            <li>
              Inputs are processed for the prediction only and kept to this local session;
              no account, no upload to third parties in this build.
            </li>
            <li>
              Models can carry bias from their training data; performance may vary across
              age, sex, language and recording conditions.
            </li>
            <li>
              Risk bands always pair colour with text, icon and pattern for colour-blind
              accessibility.
            </li>
          </ul>
        </div>
      </div>
      <p style={{ fontSize: 12.5, color: "#9AA6A3", margin: "24px 0 0" }}>
        Decision support, not diagnosis. If you or your patient have urgent symptoms, seek
        medical care directly.
      </p>
    </section>
  );
}
