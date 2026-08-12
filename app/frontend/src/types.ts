export type Mode = "voice" | "mri" | "combined";
export type RiskBand = "low" | "moderate" | "high";
export type Screen = "home" | "input" | "processing" | "results" | "error" | "about";

export interface Feature {
  name: string;
  weight: number; // 0..1 relative SHAP contribution (bar length)
  dir: "up" | "down"; // "up" raises risk, "down" lowers it
}

export interface Explanation {
  features: Feature[];
  plain_text: string;
  method: string; // "SHAP" | "Grad-CAM"
}

export interface Recommendation {
  band: string;
  route: string;
  specialist_type: string;
  secondary: string;
  disclaimer: string;
}

/** A voice take already analysed this session, kept in memory only so it can be
 *  re-run and its result compared. Nothing is stored beyond the page. */
export interface PastTake {
  url: string;
  name: string;
  blob: Blob;
  pct: number; // whole percentage the interface showed for it
  band: RiskBand;
  at: string; // wall-clock label, e.g. "14:32"
}

export interface CombinedResult {
  modality: string;
  probability: number; // 0..1
  risk_band: RiskBand;
  explanation: Explanation;
  recommendation: Recommendation;
  caveat: string;
  gradcam: string; // data-URL PNG of the Grad-CAM overlay (MRI/combined)
}
