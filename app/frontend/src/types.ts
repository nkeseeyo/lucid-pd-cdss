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

export interface CombinedResult {
  modality: string;
  probability: number; // 0..1
  risk_band: RiskBand;
  explanation: Explanation;
  recommendation: Recommendation;
  caveat: string;
  gradcam: string; // data-URL PNG of the Grad-CAM overlay (MRI/combined)
}
