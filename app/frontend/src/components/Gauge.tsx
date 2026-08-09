import type { RiskBand } from "../types";
import { bandColor, bandTint, bandLabel } from "../theme";

interface Props {
  probability: number; // 0..1
  band: RiskBand;
}

/** Arc geometry, ported from the design's `arcPath` / marker math. */
const CX = 140;
const CY = 140;
const R = 104;

function arcPath(p0: number, p1: number): string {
  const pt = (p: number): [number, number] => {
    const a = Math.PI * (1 - p);
    return [CX + R * Math.cos(a), CY - R * Math.sin(a)];
  };
  const [x0, y0] = pt(p0);
  const [x1, y1] = pt(p1);
  return `M ${x0.toFixed(1)} ${y0.toFixed(1)} A ${R} ${R} 0 0 1 ${x1.toFixed(1)} ${y1.toFixed(1)}`;
}

export default function Gauge({ probability, band }: Props) {
  const p = probability;
  // a screening estimate should never assert absolute certainty: clamp the displayed
  // percentage to 1-99 so the gauge never reads 0% or 100%.
  // `_display_pct` in app/backend/inference.py applies this same rule to the percentage
  // quoted in the plain-language summary; change one and the two numbers disagree on screen.
  const pct = Math.min(99, Math.max(1, Math.round(p * 100)));
  const color = bandColor(band);
  const a = Math.PI * (1 - p);
  const markerX = (CX + R * Math.cos(a)).toFixed(1);
  const markerY = (CY - R * Math.sin(a)).toFixed(1);
  const gaugeAria = `Risk band ${bandLabel(band)}, model probability ${pct} percent`;

  return (
    <div
      style={{
        background: "#fff",
        border: "1px solid #EAE3D5",
        borderRadius: 16,
        boxShadow: "0 1px 2px rgba(20,40,40,.04)",
        padding: 24,
      }}
    >
      <h2 style={{ fontSize: 15, fontWeight: 700, margin: "0 0 6px" }}>Estimated risk</h2>

      <div role="img" aria-label={gaugeAria} style={{ marginTop: 6 }}>
        <div style={{ position: "relative", maxWidth: 320, margin: "8px auto 0" }}>
          <svg viewBox="0 0 280 156" width="100%" style={{ display: "block" }}>
            <path d={arcPath(0, 0.4)} fill="none" stroke="#0A6E5C" strokeWidth="17"
                  strokeLinecap="round" />
            <path d={arcPath(0.4, 0.7)} fill="none" stroke="#9A5B00" strokeWidth="17"
                  strokeLinecap="round" strokeDasharray="12 9" />
            <path d={arcPath(0.7, 1)} fill="none" stroke="#A6361A" strokeWidth="17"
                  strokeLinecap="round" strokeDasharray="1.5 9" />
            <line x1="140" y1="140" x2={markerX} y2={markerY} stroke="#15302E"
                  strokeWidth="2.5" strokeLinecap="round" />
            <circle cx="140" cy="140" r="5.5" fill="#15302E" />
            <circle cx={markerX} cy={markerY} r="9" fill="#fff" stroke="#15302E"
                    strokeWidth="3.5" />
          </svg>
          <div style={{ textAlign: "center", marginTop: 4 }}>
            <div style={{ fontSize: 44, fontWeight: 700, letterSpacing: "-.02em",
                          lineHeight: 1, color }}>
              {pct}
              <span style={{ fontSize: 22, color: "#9AA6A3" }}>%</span>
            </div>
          </div>
        </div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: 8, marginTop: 14 }}>
          <ArcLegend color="#0A6E5C" label="Low" range="0–40%" icon={
            <>
              <circle cx="12" cy="12" r="9" stroke="#0A6E5C" strokeWidth="2" />
              <path d="M8 12l3 3 5-6" stroke="#0A6E5C" strokeWidth="2"
                    strokeLinecap="round" strokeLinejoin="round" />
            </>
          } />
          <ArcLegend color="#9A5B00" label="Moderate" range="40–70%" icon={
            <>
              <path d="M12 3 22 20H2L12 3Z" stroke="#9A5B00" strokeWidth="2"
                    strokeLinejoin="round" />
              <path d="M12 10v3.5" stroke="#9A5B00" strokeWidth="2" strokeLinecap="round" />
              <circle cx="12" cy="16.6" r="1" fill="#9A5B00" />
            </>
          } />
          <ArcLegend color="#A6361A" label="High" range="70–100%" icon={
            <>
              <path d="M8 3h8l5 5v8l-5 5H8l-5-5V8l5-5Z" stroke="#A6361A" strokeWidth="2"
                    strokeLinejoin="round" />
              <path d="M12 8v4.5" stroke="#A6361A" strokeWidth="2" strokeLinecap="round" />
              <circle cx="12" cy="16" r="1" fill="#A6361A" />
            </>
          } />
        </div>
      </div>

      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          marginTop: 18,
          paddingTop: 16,
          borderTop: "1px solid #EFE9DD",
        }}
      >
        <span
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 7,
            fontSize: 14,
            fontWeight: 700,
            color,
            background: bandTint(band),
            padding: "7px 13px",
            borderRadius: 999,
          }}
        >
          {bandLabel(band)} risk
        </span>
        <span style={{ fontSize: 13, color: "#7A8784" }}>
          {band === "low"
            ? "within the low band"
            : band === "moderate"
              ? "within the moderate band"
              : "within the high band"}
        </span>
      </div>
    </div>
  );
}

function ArcLegend({
  color,
  label,
  range,
  icon,
}: {
  color: string;
  label: string;
  range: string;
  icon: React.ReactNode;
}) {
  return (
    <div style={{ flex: 1, textAlign: "center" }}>
      <div
        aria-hidden="true"
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 5,
          fontSize: 12,
          fontWeight: 600,
          color,
        }}
      >
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none">
          {icon}
        </svg>
        {label}
      </div>
      <div
        style={{
          fontFamily: "'IBM Plex Mono',monospace",
          fontSize: 10.5,
          color: "#9AA6A3",
          marginTop: 2,
        }}
      >
        {range}
      </div>
    </div>
  );
}
