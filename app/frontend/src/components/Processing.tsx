import type { Mode } from "../types";

const bar = (delay: string): React.CSSProperties => ({
  width: 4,
  height: 22,
  background: "#0C5C5E",
  borderRadius: 2,
  animation: `nvbar 1s ease-in-out ${delay} infinite`,
});

export default function Processing({ mode }: { mode: Mode }) {
  const title =
    mode === "voice"
      ? "Analysing voice sample"
      : mode === "mri"
        ? "Analysing MRI"
        : "Analysing combined inputs";

  return (
    <section
      aria-busy="true"
      aria-label="Analysing"
      className="nv-screen"
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        textAlign: "center",
        minHeight: "54vh",
      }}
    >
      <div
        aria-hidden="true"
        style={{ position: "relative", width: 96, height: 96, marginBottom: 30 }}
      >
        <span
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            border: "3px solid #DCEAE7",
          }}
        />
        <span
          style={{
            position: "absolute",
            inset: 0,
            borderRadius: "50%",
            border: "3px solid transparent",
            borderTopColor: "#0C5C5E",
            animation: "nvspin 1s linear infinite",
          }}
        />
        <span
          style={{
            position: "absolute",
            inset: 0,
            display: "flex",
            alignItems: "flex-end",
            justifyContent: "center",
            gap: 4,
            paddingBottom: 38,
          }}
        >
          <span style={bar("0s")} />
          <span style={bar(".15s")} />
          <span style={bar(".3s")} />
        </span>
      </div>
      <h1 style={{ fontSize: 23, fontWeight: 700, margin: "0 0 8px" }}>{title}</h1>
      <p
        aria-live="polite"
        style={{ fontSize: 15, color: "#5A6A68", margin: 0, maxWidth: "34ch" }}
      >
        Extracting features and scoring the estimate. This usually takes a few
        seconds.
      </p>
    </section>
  );
}
