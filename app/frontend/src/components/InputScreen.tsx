import { useEffect, useRef, useState } from "react";
import type { Mode } from "../types";

type InputState = "empty" | "recording" | "recorded";

interface VoiceFile {
  url: string;
  name: string;
  blob: Blob;
}
interface MriFile {
  url: string;
  name: string;
  blob: Blob;
}

interface Props {
  mode: Mode;
  voice: VoiceFile | null;
  mri: MriFile | null;
  onMode: (m: Mode) => void;
  onVoice: (v: VoiceFile | null) => void;
  onMri: (m: MriFile | null) => void;
  onAnalyse: () => void;
  onHome: () => void;
  /** Bumped by the parent whenever we leave the input screen, so the recorder
   *  tears down its mic/AudioContext/rAF even mid-recording. */
  teardownSignal: number;
}

const MAX_BYTES = 25 * 1024 * 1024;

export default function InputScreen({
  mode,
  voice,
  mri,
  onMode,
  onVoice,
  onMri,
  onAnalyse,
  onHome,
  teardownSignal,
}: Props) {
  const [inputState, setInputState] = useState<InputState>(
    voice ? "recorded" : "empty",
  );
  const [recordSeconds, setRecordSeconds] = useState(0);
  const [error, setError] = useState<string | null>(null);

  const waveformRef = useRef<HTMLCanvasElement | null>(null);
  const audioInputRef = useRef<HTMLInputElement | null>(null);
  const mriInputRef = useRef<HTMLInputElement | null>(null);

  // imperative recorder handles (not state, they don't drive render)
  const streamRef = useRef<MediaStream | null>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);
  const rafRef = useRef<number | null>(null);
  const timerRef = useRef<number | null>(null);
  const t0Ref = useRef<number>(0);

  function cleanupMic() {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    try {
      if (recorderRef.current && recorderRef.current.state !== "inactive")
        recorderRef.current.stop();
    } catch {
      /* ignore */
    }
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {
      /* ignore */
    }
    try {
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed")
        void audioCtxRef.current.close();
    } catch {
      /* ignore */
    }
    recorderRef.current = null;
    streamRef.current = null;
    audioCtxRef.current = null;
    analyserRef.current = null;
  }

  // Tear down on unmount and whenever the parent signals we are leaving input.
  useEffect(() => {
    return () => cleanupMic();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (teardownSignal > 0) cleanupMic();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [teardownSignal]);

  function drawWave() {
    const cv = waveformRef.current;
    const analyser = analyserRef.current;
    if (!cv || !analyser) return;
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    const buf = new Uint8Array(analyser.fftSize);
    const loop = () => {
      const a = analyserRef.current;
      if (!a) return;
      a.getByteTimeDomainData(buf);
      ctx.clearRect(0, 0, cv.width, cv.height);
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = "#0C5C5E";
      ctx.beginPath();
      const slice = cv.width / buf.length;
      for (let i = 0; i < buf.length; i++) {
        const v = buf[i] / 128.0;
        const y = (v * cv.height) / 2;
        const x = i * slice;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      rafRef.current = requestAnimationFrame(loop);
    };
    loop();
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      streamRef.current = stream;
      const Ctx: typeof AudioContext =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
      const audioCtx = new Ctx();
      audioCtxRef.current = audioCtx;
      const src = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 1024;
      src.connect(analyser);
      analyserRef.current = analyser;

      chunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      recorderRef.current = recorder;
      recorder.ondataavailable = (e) => {
        if (e.data && e.data.size) chunksRef.current.push(e.data);
      };
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: "audio/webm" });
        const url = URL.createObjectURL(blob);
        onVoice({ url, name: "live-recording.webm", blob });
        setInputState("recorded");
      };
      recorder.start();
      setInputState("recording");
      setRecordSeconds(0);
      setError(null);
      t0Ref.current = Date.now();
      timerRef.current = window.setInterval(() => {
        const s = (Date.now() - t0Ref.current) / 1000;
        setRecordSeconds(s);
        if (s >= 20) stopRecording();
      }, 100);
      drawWave();
    } catch {
      setError(
        "Microphone access was blocked or is unavailable. You can upload an audio file instead.",
      );
      setInputState("empty");
    }
  }

  function stopRecording() {
    if (timerRef.current !== null) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
    try {
      if (recorderRef.current && recorderRef.current.state !== "inactive")
        recorderRef.current.stop();
    } catch {
      /* ignore */
    }
    try {
      streamRef.current?.getTracks().forEach((t) => t.stop());
    } catch {
      /* ignore */
    }
    try {
      if (audioCtxRef.current && audioCtxRef.current.state !== "closed")
        void audioCtxRef.current.close();
    } catch {
      /* ignore */
    }
    analyserRef.current = null;
  }

  function reRecord() {
    cleanupMic();
    onVoice(null);
    setInputState("empty");
    setError(null);
  }

  // ---- file handling ----
  const pickAudio = () => audioInputRef.current?.click();
  const pickMri = () => mriInputRef.current?.click();
  const onDragOver = (e: React.DragEvent) => e.preventDefault();

  function handleAudio(file: File | undefined | null) {
    if (!file) return;
    if (!file.type.startsWith("audio/")) {
      setError("That file isn’t audio. Please upload a WAV, MP3 or M4A clip.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setError("That file is over 25 MB. Please use a shorter clip.");
      return;
    }
    const url = URL.createObjectURL(file);
    onVoice({ url, name: file.name, blob: file });
    setInputState("recorded");
    setError(null);
  }
  function handleMri(file: File | undefined | null) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("That file isn’t an image. Please upload a JPG or PNG export.");
      return;
    }
    if (file.size > MAX_BYTES) {
      setError("That image is over 25 MB. Please use a smaller file.");
      return;
    }
    const url = URL.createObjectURL(file);
    onMri({ url, name: file.name, blob: file });
    setError(null);
  }
  function removeMri() {
    onMri(null);
    pickMri();
  }

  // ---- derived ----
  const showVoiceInput = mode === "voice" || mode === "combined";
  const showMriInput = mode === "mri" || mode === "combined";
  const canAnalyse =
    mode === "voice"
      ? !!voice
      : mode === "mri"
        ? !!mri
        : !!voice && !!mri;

  const showModeCaveat = mode !== "voice";
  const modeCaveatText =
    mode === "mri"
      ? "Research baseline. MRI results are exploratory and must not guide care on their own, so interpret with caution."
      : "Illustrative only. Voice and MRI come from different sources, so this is not a true fused prediction.";
  const caveatCol =
    mode === "mri"
      ? { bg: "#F2EEF8", border: "#E0D6F0", fg: "#5B3E86" }
      : { bg: "#F2EEE2", border: "#E4DAC2", fg: "#7A5E22" };

  const readyHint = canAnalyse
    ? "Input ready. Results are an estimate, not a diagnosis."
    : mode === "combined"
      ? "Add both a voice sample and an MRI to continue."
      : mode === "mri"
        ? "Add an MRI image to continue."
        : "Record or upload a voice sample to continue.";

  function switchMode(m: Mode) {
    cleanupMic();
    setInputState("empty");
    setError(null);
    setRecordSeconds(0);
    onMode(m);
  }

  // ---- tab styling ----
  const tabBg = (m: Mode) => (mode === m ? "#fff" : "transparent");
  const tabFg = (m: Mode) => (mode === m ? "#0C5C5E" : "#4A5A58");
  const tabSh = (m: Mode) => (mode === m ? "0 1px 2px rgba(20,40,40,.12)" : "none");
  const tab = (
    m: Mode,
    label: string,
    badgeText: string,
    badgeFg: string,
    badgeBg: string,
  ) => (
    <button
      role="tab"
      data-nv-focusring
      aria-selected={mode === m}
      onClick={() => switchMode(m)}
      style={{
        border: "none",
        cursor: "pointer",
        borderRadius: 9,
        padding: "9px 16px",
        font: "600 14.5px 'IBM Plex Sans'",
        minHeight: 42,
        display: "flex",
        alignItems: "center",
        gap: 8,
        background: tabBg(m),
        color: tabFg(m),
        boxShadow: tabSh(m),
      }}
    >
      {label}{" "}
      <span
        style={{
          fontSize: 10.5,
          fontWeight: 500,
          color: badgeFg,
          background: badgeBg,
          padding: "2px 6px",
          borderRadius: 6,
        }}
      >
        {badgeText}
      </span>
    </button>
  );

  return (
    <section aria-labelledby="nv-input-h" className="nv-screen" style={{ opacity: 1 }}>
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
        id="nv-input-h"
        style={{
          fontSize: 28,
          fontWeight: 700,
          letterSpacing: "-.01em",
          margin: "0 0 6px",
        }}
      >
        New screening
      </h1>
      <p style={{ fontSize: 15.5, color: "#5A6A68", margin: "0 0 20px" }}>
        Choose a mode, add your input, then analyse.
      </p>

      <div
        role="tablist"
        aria-label="Prediction mode"
        style={{
          display: "inline-flex",
          background: "#EAE4D7",
          borderRadius: 12,
          padding: 4,
          gap: 4,
          marginBottom: 24,
          flexWrap: "wrap",
        }}
      >
        {tab("voice", "Voice", "Primary", "#0A6E5C", "#E3F0EC")}
        {tab("mri", "MRI", "Research", "#6A4F8C", "#EEEAF3")}
        {tab("combined", "Combined", "Illustrative", "#8A6A2A", "#F0ECE2")}
      </div>

      {showModeCaveat && (
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
            marginBottom: 22,
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
            {modeCaveatText}
          </p>
        </div>
      )}

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(0,1fr)",
          gap: 18,
          maxWidth: 760,
        }}
      >
        {/* VOICE recorder card */}
        {showVoiceInput && (
          <div
            style={{
              background: "#fff",
              border: "1px solid #EAE3D5",
              borderRadius: 16,
              boxShadow: "0 1px 2px rgba(20,40,40,.04)",
              padding: 26,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 4,
              }}
            >
              <h2 style={{ fontSize: 17, fontWeight: 700, margin: 0 }}>Voice sample</h2>
              <span
                style={{
                  fontFamily: "'IBM Plex Mono',monospace",
                  fontSize: 11,
                  color: "#9AA6A3",
                }}
              >
                WAV · MP3 · M4A · ≤25 MB
              </span>
            </div>
            <p style={{ fontSize: 14, color: "#5A6A68", margin: "6px 0 20px" }}>
              Take a breath, then hold a steady &ldquo;aaah&rdquo; for at least 5 seconds,
              as long as is comfortable. Recording stops by itself at 20.
            </p>

            {inputState === "empty" && (
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  alignItems: "center",
                  gap: 16,
                  padding: "14px 0 6px",
                }}
              >
                <button
                  data-nv-focusring
                  onClick={startRecording}
                  aria-label="Start recording"
                  onMouseEnter={(e) => (e.currentTarget.style.background = "#084244")}
                  onMouseLeave={(e) => (e.currentTarget.style.background = "#0C5C5E")}
                  style={{
                    position: "relative",
                    width: 84,
                    height: 84,
                    borderRadius: "50%",
                    background: "#0C5C5E",
                    border: "none",
                    cursor: "pointer",
                    display: "grid",
                    placeItems: "center",
                  }}
                >
                  <span
                    style={{
                      width: 26,
                      height: 26,
                      borderRadius: "50%",
                      background: "#fff",
                    }}
                  />
                </button>
                <span style={{ fontSize: 14, fontWeight: 600, color: "#0C5C5E" }}>
                  Tap to record
                </span>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 12,
                    width: "100%",
                    color: "#9AA6A3",
                    fontSize: 12.5,
                  }}
                >
                  <span style={{ flex: 1, height: 1, background: "#E4DDCE" }} />
                  or
                  <span style={{ flex: 1, height: 1, background: "#E4DDCE" }} />
                </div>
                <div
                  tabIndex={0}
                  role="button"
                  data-nv-focusring
                  aria-label="Upload an audio file"
                  onClick={pickAudio}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      pickAudio();
                    }
                  }}
                  onDragOver={onDragOver}
                  onDrop={(e) => {
                    e.preventDefault();
                    handleAudio(e.dataTransfer.files?.[0]);
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = "#0C5C5E";
                    e.currentTarget.style.background = "#F4F8F7";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = "#C7D6D3";
                    e.currentTarget.style.background = "#FAFBFA";
                  }}
                  style={{
                    width: "100%",
                    border: "1.5px dashed #C7D6D3",
                    borderRadius: 12,
                    padding: 22,
                    textAlign: "center",
                    cursor: "pointer",
                    background: "#FAFBFA",
                  }}
                >
                  <svg
                    width="24"
                    height="24"
                    viewBox="0 0 24 24"
                    fill="none"
                    aria-hidden="true"
                    style={{ marginBottom: 6 }}
                  >
                    <path
                      d="M12 16V4m0 0L7 9m5-5 5 5"
                      stroke="#0C5C5E"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                    <path
                      d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"
                      stroke="#0C5C5E"
                      strokeWidth="2"
                      strokeLinecap="round"
                    />
                  </svg>
                  <div style={{ fontSize: 14, fontWeight: 600, color: "#15302E" }}>
                    Drag a clip here, or{" "}
                    <span style={{ color: "#0C5C5E", textDecoration: "underline" }}>
                      browse
                    </span>
                  </div>
                </div>
              </div>
            )}

            {inputState === "recording" && (
              <div style={{ padding: "6px 0" }}>
                <div
                  style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: 12,
                  }}
                >
                  <span
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 9,
                      fontSize: 14,
                      fontWeight: 600,
                      color: "#A6361A",
                    }}
                  >
                    <span
                      aria-hidden="true"
                      style={{ position: "relative", width: 12, height: 12 }}
                    >
                      <span
                        style={{
                          position: "absolute",
                          inset: 0,
                          borderRadius: "50%",
                          background: "#A6361A",
                        }}
                      />
                      <span
                        style={{
                          position: "absolute",
                          inset: 0,
                          borderRadius: "50%",
                          background: "#A6361A",
                          animation: "nvpulse 1.4s ease-out infinite",
                        }}
                      />
                    </span>
                    Recording…
                  </span>
                  <span
                    aria-live="polite"
                    style={{
                      fontFamily: "'IBM Plex Mono',monospace",
                      fontSize: 15,
                      fontWeight: 500,
                      color: "#15302E",
                    }}
                  >
                    {recordSeconds.toFixed(1)}s
                  </span>
                </div>
                <canvas
                  ref={waveformRef}
                  width={640}
                  height={84}
                  aria-hidden="true"
                  style={{
                    width: "100%",
                    height: 84,
                    background: "#F4F8F7",
                    borderRadius: 10,
                    border: "1px solid #E0EAE8",
                  }}
                />
                <div
                  style={{ display: "flex", justifyContent: "center", marginTop: 16 }}
                >
                  <button
                    data-nv-focusring
                    onClick={stopRecording}
                    style={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 9,
                      background: "#15302E",
                      color: "#fff",
                      border: "none",
                      borderRadius: 11,
                      height: 48,
                      padding: "0 22px",
                      font: "600 15px 'IBM Plex Sans'",
                      cursor: "pointer",
                    }}
                  >
                    <span
                      aria-hidden="true"
                      style={{
                        width: 14,
                        height: 14,
                        background: "#fff",
                        borderRadius: 3,
                      }}
                    />
                    Stop &amp; keep
                  </button>
                </div>
              </div>
            )}

            {inputState === "recorded" && voice && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  background: "#F1F6F5",
                  border: "1px solid #DCEAE7",
                  borderRadius: 12,
                  padding: "16px 18px",
                }}
              >
                <span
                  aria-hidden="true"
                  style={{
                    width: 38,
                    height: 38,
                    borderRadius: 10,
                    background: "#0C5C5E",
                    display: "grid",
                    placeItems: "center",
                    flex: "none",
                  }}
                >
                  <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
                    <path
                      d="M5 13l4 4L19 7"
                      stroke="#fff"
                      strokeWidth="2.4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  </svg>
                </span>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 14.5,
                      fontWeight: 600,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {voice.name}
                  </div>
                  <audio
                    controls
                    src={voice.url}
                    style={{ width: "100%", height: 34, marginTop: 8 }}
                  />
                </div>
                <button
                  data-nv-focusring
                  onClick={reRecord}
                  style={{
                    background: "#fff",
                    border: "1.5px solid #CADEDB",
                    borderRadius: 9,
                    height: 40,
                    padding: "0 14px",
                    font: "600 13.5px 'IBM Plex Sans'",
                    color: "#0C5C5E",
                    cursor: "pointer",
                    flex: "none",
                  }}
                >
                  Re-record
                </button>
              </div>
            )}
          </div>
        )}

        {/* MRI uploader card */}
        {showMriInput && (
          <div
            style={{
              background: "#fff",
              border: "1px solid #EAE3D5",
              borderRadius: 16,
              boxShadow: "0 1px 2px rgba(20,40,40,.04)",
              padding: 26,
            }}
          >
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                marginBottom: 4,
              }}
            >
              <h2 style={{ fontSize: 17, fontWeight: 700, margin: 0 }}>Brain MRI</h2>
              <span
                style={{
                  fontFamily: "'IBM Plex Mono',monospace",
                  fontSize: 11,
                  color: "#9AA6A3",
                }}
              >
                JPG · PNG · ≤25 MB
              </span>
            </div>
            <p style={{ fontSize: 14, color: "#5A6A68", margin: "6px 0 18px" }}>
              Upload an axial T1/T2 slice. Used as a research baseline only.
            </p>

            {!mri && (
              <div
                tabIndex={0}
                role="button"
                data-nv-focusring
                aria-label="Upload an MRI image"
                onClick={pickMri}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    pickMri();
                  }
                }}
                onDragOver={onDragOver}
                onDrop={(e) => {
                  e.preventDefault();
                  handleMri(e.dataTransfer.files?.[0]);
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.borderColor = "#6A4F8C";
                  e.currentTarget.style.background = "#F7F5FB";
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.borderColor = "#C7D6D3";
                  e.currentTarget.style.background = "#FAFBFA";
                }}
                style={{
                  border: "1.5px dashed #C7D6D3",
                  borderRadius: 12,
                  padding: "34px 22px",
                  textAlign: "center",
                  cursor: "pointer",
                  background: "#FAFBFA",
                }}
              >
                <svg
                  width="26"
                  height="26"
                  viewBox="0 0 24 24"
                  fill="none"
                  aria-hidden="true"
                  style={{ marginBottom: 8 }}
                >
                  <rect x="3" y="4" width="18" height="16" rx="2" stroke="#6A4F8C" strokeWidth="2" />
                  <path
                    d="M3 16l5-4 4 3 3-2 6 5"
                    stroke="#6A4F8C"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                  <circle cx="9" cy="9" r="1.6" fill="#6A4F8C" />
                </svg>
                <div style={{ fontSize: 14.5, fontWeight: 600, color: "#15302E" }}>
                  Drag an MRI here, or{" "}
                  <span style={{ color: "#6A4F8C", textDecoration: "underline" }}>
                    browse
                  </span>
                </div>
              </div>
            )}

            {mri && (
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  background: "#F7F5FB",
                  border: "1px solid #E6DFF1",
                  borderRadius: 12,
                  padding: "14px 16px",
                }}
              >
                <img
                  src={mri.url}
                  alt="Uploaded MRI preview"
                  style={{
                    width: 56,
                    height: 56,
                    objectFit: "cover",
                    borderRadius: 9,
                    background: "#15302E",
                    flex: "none",
                  }}
                />
                <div style={{ flex: 1, minWidth: 0 }}>
                  <div
                    style={{
                      fontSize: 14.5,
                      fontWeight: 600,
                      whiteSpace: "nowrap",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                    }}
                  >
                    {mri.name}
                  </div>
                  <div style={{ fontSize: 12.5, color: "#7A6A8C", marginTop: 2 }}>
                    Ready for analysis
                  </div>
                </div>
                <button
                  data-nv-focusring
                  onClick={removeMri}
                  style={{
                    background: "#fff",
                    border: "1.5px solid #D8CCE8",
                    borderRadius: 9,
                    height: 40,
                    padding: "0 14px",
                    font: "600 13.5px 'IBM Plex Sans'",
                    color: "#6A4F8C",
                    cursor: "pointer",
                    flex: "none",
                  }}
                >
                  Replace
                </button>
              </div>
            )}
          </div>
        )}

        {/* inline error */}
        {error && (
          <div
            role="alert"
            style={{
              display: "flex",
              gap: 11,
              alignItems: "flex-start",
              background: "#F8E6DF",
              border: "1px solid #E9C4B6",
              borderRadius: 12,
              padding: "14px 16px",
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
              <circle cx="12" cy="12" r="9" stroke="#A6361A" strokeWidth="2" />
              <path d="M12 7v6" stroke="#A6361A" strokeWidth="2" strokeLinecap="round" />
              <circle cx="12" cy="16.4" r="1.1" fill="#A6361A" />
            </svg>
            <p style={{ margin: 0, fontSize: 14, lineHeight: 1.5, color: "#8A2A12" }}>
              {error}
            </p>
          </div>
        )}

        {/* analyse bar */}
        <div
          style={{
            display: "flex",
            flexWrap: "wrap",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 14,
            paddingTop: 4,
          }}
        >
          <p
            style={{
              margin: 0,
              fontSize: 13.5,
              color: "#7A8784",
              maxWidth: "42ch",
            }}
          >
            {readyHint}
          </p>
          {canAnalyse ? (
            <button
              data-nv-focusring
              onClick={onAnalyse}
              onMouseEnter={(e) => (e.currentTarget.style.background = "#084244")}
              onMouseLeave={(e) => (e.currentTarget.style.background = "#0C5C5E")}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 9,
                background: "#0C5C5E",
                color: "#fff",
                border: "none",
                borderRadius: 11,
                height: 52,
                padding: "0 26px",
                font: "600 16px 'IBM Plex Sans'",
                cursor: "pointer",
              }}
            >
              Analyse
              <svg
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                aria-hidden="true"
              >
                <path
                  d="M5 12h13M13 6l6 6-6 6"
                  stroke="#fff"
                  strokeWidth="2.2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
            </button>
          ) : (
            <button
              disabled
              aria-disabled="true"
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 9,
                background: "#DCDDD6",
                color: "#9AA29E",
                border: "none",
                borderRadius: 11,
                height: 52,
                padding: "0 26px",
                font: "600 16px 'IBM Plex Sans'",
                cursor: "not-allowed",
              }}
            >
              Analyse
            </button>
          )}
        </div>
      </div>

      <input
        ref={audioInputRef}
        type="file"
        accept="audio/*"
        onChange={(e) => handleAudio(e.target.files?.[0])}
        style={{ display: "none" }}
        aria-hidden="true"
      />
      <input
        ref={mriInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleMri(e.target.files?.[0])}
        style={{ display: "none" }}
        aria-hidden="true"
      />
    </section>
  );
}
