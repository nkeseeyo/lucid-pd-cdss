import type { CombinedResult, Mode } from "./types";

/**
 * Backend origin. VITE_API_URL overrides it; otherwise the dev server talks to the
 * local FastAPI process, while a production build uses the page's own origin, because
 * the deployed container serves this bundle and the API together.
 */
const API: string =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/+$/, "") ??
  (import.meta.env.DEV ? "http://127.0.0.1:8000" : "");

/**
 * POST the captured input(s) to the FastAPI backend and return the structured
 * result. Throws on any non-OK response or network failure — the caller routes
 * to the error screen (no silent mock fallback).
 */
export async function predict(
  mode: Mode,
  files: { voice?: Blob | null; image?: Blob | null },
): Promise<CombinedResult> {
  const fd = new FormData();
  let path: string;
  if (mode === "voice") {
    fd.append("file", files.voice as Blob, "voice.webm");
    path = "/predict/voice";
  } else if (mode === "mri") {
    fd.append("file", files.image as Blob, "mri.png");
    path = "/predict/mri";
  } else {
    fd.append("voice", files.voice as Blob, "voice.webm");
    fd.append("image", files.image as Blob, "mri.png");
    path = "/predict/combined";
  }
  const res = await fetch(`${API}${path}`, { method: "POST", body: fd });
  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    throw new Error(`Request failed (${res.status}). ${detail}`.trim());
  }
  return (await res.json()) as CombinedResult;
}
