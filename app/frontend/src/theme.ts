import type { RiskBand } from "./types";

/** Band colour helpers, pulled directly from the design reference palette. */
export const bandColor = (b: RiskBand): string =>
  b === "low" ? "#0A6E5C" : b === "moderate" ? "#9A5B00" : "#A6361A";

export const bandTint = (b: RiskBand): string =>
  b === "low" ? "#E3F0EC" : b === "moderate" ? "#F6EAD6" : "#F6E2DA";

export const bandLabel = (b: RiskBand): string =>
  b === "low" ? "Low" : b === "moderate" ? "Moderate" : "High";
