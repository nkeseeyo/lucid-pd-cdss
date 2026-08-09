interface Props {
  onStart: () => void;
  onAbout: () => void;
  onVoice: () => void;
  onMri: () => void;
  onCombined: () => void;
}

const ArrowWhite = () => (
  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M5 12h13M13 6l6 6-6 6"
      stroke="#fff"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);
const ArrowWhiteSm = () => (
  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
    <path
      d="M5 12h13M13 6l6 6-6 6"
      stroke="#fff"
      strokeWidth="2.2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

const cardStyle: React.CSSProperties = {
  display: "flex",
  flexDirection: "column",
  background: "#fff",
  border: "1px solid #EAE3D5",
  borderRadius: 16,
  boxShadow: "0 1px 2px rgba(20,40,40,.04)",
  padding: 24,
};
const badge = (fg: string, bg: string): React.CSSProperties => ({
  display: "inline-flex",
  alignItems: "center",
  gap: 5,
  fontSize: 11.5,
  fontWeight: 600,
  color: fg,
  background: bg,
  padding: "3px 9px",
  borderRadius: 999,
});
const cardP: React.CSSProperties = {
  fontSize: 14.5,
  lineHeight: 1.5,
  color: "#4A5A58",
  margin: "0 0 20px",
  flex: 1,
};
const outlineBtn: React.CSSProperties = {
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 8,
  background: "#fff",
  color: "#0C5C5E",
  border: "1.5px solid #CADEDB",
  borderRadius: 10,
  height: 46,
  font: "600 15px 'IBM Plex Sans'",
  cursor: "pointer",
};
const step = (n: string, title: string, body: string) => (
  <div style={{ display: "flex", gap: 13 }}>
    <span
      aria-hidden="true"
      style={{
        fontFamily: "'IBM Plex Mono',monospace",
        fontSize: 13,
        fontWeight: 500,
        color: "#0C5C5E",
        background: "#E6F0EE",
        width: 28,
        height: 28,
        borderRadius: 8,
        display: "grid",
        placeItems: "center",
        flex: "none",
      }}
    >
      {n}
    </span>
    <div>
      <div style={{ fontWeight: 600, fontSize: 15, marginBottom: 3 }}>{title}</div>
      <p style={{ margin: 0, fontSize: 13.5, color: "#5A6A68", lineHeight: 1.5 }}>
        {body}
      </p>
    </div>
  </div>
);

export default function Home({ onStart, onAbout, onVoice, onMri, onCombined }: Props) {
  const enter = (e: React.MouseEvent<HTMLButtonElement>, c: string) =>
    (e.currentTarget.style.background = c);
  const enterBorder = (e: React.MouseEvent<HTMLButtonElement>, c: string) =>
    (e.currentTarget.style.borderColor = c);

  return (
    <section aria-labelledby="nv-home-h" className="nv-screen" style={{ opacity: 1 }}>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0,1.15fr) minmax(0,.85fr)",
          gap: 48,
          alignItems: "center",
          padding: "18px 0 8px",
        }}
      >
        <div>
          <div
            style={{
              fontFamily: "'IBM Plex Mono',monospace",
              fontSize: 12.5,
              letterSpacing: ".12em",
              textTransform: "uppercase",
              color: "#0C5C5E",
              fontWeight: 500,
              marginBottom: 14,
            }}
          >
            Clinical decision support
          </div>
          <h1
            id="nv-home-h"
            style={{
              fontSize: 42,
              lineHeight: 1.08,
              letterSpacing: "-.02em",
              fontWeight: 700,
              margin: "0 0 16px",
              textWrap: "balance",
            }}
          >
            A transparent second opinion on Parkinson&rsquo;s risk
          </h1>
          <p
            style={{
              fontSize: 17.5,
              lineHeight: 1.55,
              color: "#3C4D4B",
              margin: "0 0 26px",
              maxWidth: "38ch",
            }}
          >
            Record a short sustained vowel and get an explainable risk estimate with a
            suggested care route — built for clarity under time pressure.
          </p>
          <div
            style={{ display: "flex", flexWrap: "wrap", gap: 12, alignItems: "center" }}
          >
            <button
              data-nv-focusring
              onClick={onStart}
              onMouseEnter={(e) => enter(e, "#084244")}
              onMouseLeave={(e) => enter(e, "#0C5C5E")}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 9,
                background: "#0C5C5E",
                color: "#fff",
                border: "none",
                borderRadius: 11,
                padding: "0 24px",
                height: 52,
                font: "600 16.5px 'IBM Plex Sans'",
                cursor: "pointer",
              }}
            >
              Start a screening
              <ArrowWhite />
            </button>
            <button
              data-nv-focusring
              onClick={onAbout}
              onMouseEnter={(e) => enterBorder(e, "#0C5C5E")}
              onMouseLeave={(e) => enterBorder(e, "#CADEDB")}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 8,
                background: "#fff",
                color: "#0C5C5E",
                border: "1.5px solid #CADEDB",
                borderRadius: 11,
                padding: "0 20px",
                height: 52,
                font: "600 15.5px 'IBM Plex Sans'",
                cursor: "pointer",
              }}
            >
              How it works
            </button>
          </div>
        </div>

        {/* Example result card */}
        <div
          aria-hidden="true"
          style={{
            position: "relative",
            background: "#fff",
            border: "1px solid #EAE3D5",
            borderRadius: 20,
            boxShadow: "0 1px 2px rgba(20,40,40,.04),0 18px 44px rgba(20,40,40,.07)",
            padding: 26,
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              marginBottom: 18,
            }}
          >
            <span
              style={{
                fontFamily: "'IBM Plex Mono',monospace",
                fontSize: 11,
                letterSpacing: ".08em",
                color: "#7A8784",
                textTransform: "uppercase",
              }}
            >
              Example result
            </span>
            <span style={badge("#9A5B00", "#F6EAD6")}>
              <svg width="13" height="13" viewBox="0 0 24 24" aria-hidden="true">
                <path
                  d="M12 3 22 20H2L12 3Z"
                  fill="none"
                  stroke="#9A5B00"
                  strokeWidth="2.2"
                  strokeLinejoin="round"
                />
                <path d="M12 10v4" stroke="#9A5B00" strokeWidth="2.2" strokeLinecap="round" />
                <circle cx="12" cy="17" r="1.1" fill="#9A5B00" />
              </svg>
              Moderate
            </span>
          </div>
          <div
            style={{
              fontSize: 46,
              fontWeight: 700,
              letterSpacing: "-.02em",
              lineHeight: 1,
            }}
          >
            64<span style={{ fontSize: 24, color: "#7A8784" }}>%</span>
          </div>
          <div style={{ fontSize: 13, color: "#7A8784", margin: "2px 0 18px" }}>
            model probability
          </div>
          <div
            style={{
              display: "flex",
              gap: 5,
              height: 9,
              borderRadius: 6,
              overflow: "hidden",
              marginBottom: 8,
            }}
          >
            <span style={{ flex: 40, background: "#0A6E5C" }} />
            <span
              style={{
                flex: 30,
                background:
                  "repeating-linear-gradient(45deg,#9A5B00,#9A5B00 3px,#b87a22 3px,#b87a22 6px)",
              }}
            />
            <span style={{ flex: 30, background: "#EBD9D2" }} />
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              fontFamily: "'IBM Plex Mono',monospace",
              fontSize: 10,
              color: "#9AA6A3",
            }}
          >
            <span>Low</span>
            <span>Moderate</span>
            <span>High</span>
          </div>
        </div>
      </div>

      <h2
        style={{
          fontSize: 14,
          letterSpacing: ".04em",
          textTransform: "uppercase",
          color: "#7A8784",
          fontWeight: 600,
          margin: "46px 0 16px",
        }}
      >
        Choose a prediction mode
      </h2>
      <div
        role="list"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3,minmax(0,1fr))",
          gap: 18,
        }}
      >
        {/* Voice */}
        <div role="listitem" style={cardStyle}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 11,
              marginBottom: 13,
            }}
          >
            <span
              aria-hidden="true"
              style={{
                width: 42,
                height: 42,
                borderRadius: 11,
                background: "#E6F0EE",
                display: "grid",
                placeItems: "center",
                flex: "none",
              }}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 3a3 3 0 0 1 3 3v6a3 3 0 0 1-6 0V6a3 3 0 0 1 3-3Z"
                  stroke="#0C5C5E"
                  strokeWidth="2"
                />
                <path
                  d="M5 11a7 7 0 0 0 14 0M12 18v3"
                  stroke="#0C5C5E"
                  strokeWidth="2"
                  strokeLinecap="round"
                />
              </svg>
            </span>
            <span style={badge("#0A6E5C", "#E3F0EC")}>Primary mode</span>
          </div>
          <h3 style={{ fontSize: 19, fontWeight: 700, margin: "0 0 6px" }}>Voice</h3>
          <p style={cardP}>
            Record a sustained &ldquo;aaah&rdquo; or upload a short clip. Acoustic
            features explained with SHAP.
          </p>
          <button
            data-nv-focusring
            onClick={onVoice}
            onMouseEnter={(e) => enter(e, "#084244")}
            onMouseLeave={(e) => enter(e, "#0C5C5E")}
            style={{
              display: "inline-flex",
              alignItems: "center",
              justifyContent: "center",
              gap: 8,
              background: "#0C5C5E",
              color: "#fff",
              border: "none",
              borderRadius: 10,
              height: 46,
              font: "600 15px 'IBM Plex Sans'",
              cursor: "pointer",
            }}
          >
            Start voice
            <ArrowWhiteSm />
          </button>
        </div>

        {/* MRI */}
        <div role="listitem" style={cardStyle}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 11,
              marginBottom: 13,
            }}
          >
            <span
              aria-hidden="true"
              style={{
                width: 42,
                height: 42,
                borderRadius: 11,
                background: "#EEEAF3",
                display: "grid",
                placeItems: "center",
                flex: "none",
              }}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <path
                  d="M12 4c4 0 7 2.8 7 6.5 0 1.6-.6 2.6-.6 4 0 1.2.6 1.6.6 2.8 0 1.4-1.2 2.2-2.6 2.2-1 0-1.4-.6-2.4-.6s-1.4.6-2 .6c-4 0-7-3.8-7-8.5C3 6.8 8 4 12 4Z"
                  stroke="#6A4F8C"
                  strokeWidth="1.8"
                  strokeLinejoin="round"
                />
                <path
                  d="M9.5 11.5h0M14 9.5h0M13 14h0"
                  stroke="#6A4F8C"
                  strokeWidth="2.4"
                  strokeLinecap="round"
                />
              </svg>
            </span>
            <span style={badge("#6A4F8C", "#EEEAF3")}>Research baseline</span>
          </div>
          <h3 style={{ fontSize: 19, fontWeight: 700, margin: "0 0 6px" }}>MRI</h3>
          <p style={cardP}>
            Upload a brain MRI. Grad-CAM highlights regions —{" "}
            <strong style={{ color: "#6A4F8C" }}>interpret with caution</strong>.
          </p>
          <button
            data-nv-focusring
            onClick={onMri}
            onMouseEnter={(e) => enterBorder(e, "#0C5C5E")}
            onMouseLeave={(e) => enterBorder(e, "#CADEDB")}
            style={outlineBtn}
          >
            Start MRI
          </button>
        </div>

        {/* Combined */}
        <div role="listitem" style={cardStyle}>
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 11,
              marginBottom: 13,
            }}
          >
            <span
              aria-hidden="true"
              style={{
                width: 42,
                height: 42,
                borderRadius: 11,
                background: "#F0ECE2",
                display: "grid",
                placeItems: "center",
                flex: "none",
              }}
            >
              <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                <circle cx="9" cy="12" r="5.5" stroke="#8A6A2A" strokeWidth="1.8" />
                <circle cx="15" cy="12" r="5.5" stroke="#8A6A2A" strokeWidth="1.8" />
              </svg>
            </span>
            <span style={badge("#8A6A2A", "#F0ECE2")}>Illustrative only</span>
          </div>
          <h3 style={{ fontSize: 19, fontWeight: 700, margin: "0 0 6px" }}>Combined</h3>
          <p style={cardP}>
            Voice + MRI together. Inputs come from different sources, so this is not true
            fusion.
          </p>
          <button
            data-nv-focusring
            onClick={onCombined}
            onMouseEnter={(e) => enterBorder(e, "#0C5C5E")}
            onMouseLeave={(e) => enterBorder(e, "#CADEDB")}
            style={outlineBtn}
          >
            Start combined
          </button>
        </div>
      </div>

      {/* How it works */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(3,1fr)",
          gap: 18,
          marginTop: 34,
          paddingTop: 30,
          borderTop: "1px solid #E4DDCE",
        }}
      >
        {step(
          "1",
          "Capture an input",
          "Record in-browser or drag in a file. Nothing leaves this session.",
        )}
        {step(
          "2",
          "Model estimates risk",
          "A model probability mapped to Low / Moderate / High bands.",
        )}
        {step(
          "3",
          "Explain & route",
          "See why, in plain language, plus a suggested care route.",
        )}
      </div>
    </section>
  );
}
