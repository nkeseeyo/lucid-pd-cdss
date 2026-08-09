import { useCallback, useRef, useState } from "react";
import type { CombinedResult, Mode, Screen } from "./types";
import { predict } from "./api";
import TopBanner from "./components/TopBanner";
import Header from "./components/Header";
import Footer from "./components/Footer";
import Home from "./components/Home";
import InputScreen from "./components/InputScreen";
import Processing from "./components/Processing";
import ErrorScreen from "./components/ErrorScreen";
import Results from "./components/Results";
import About from "./components/About";

interface CapturedFile {
  url: string;
  name: string;
  blob: Blob;
}

/** Minimum visible processing time so the spinner animation isn't a flash. */
const MIN_PROCESSING_MS = 1900;

export default function App() {
  const [screen, setScreen] = useState<Screen>("home");
  const [mode, setMode] = useState<Mode>("voice");
  const [voice, setVoice] = useState<CapturedFile | null>(null);
  const [mri, setMri] = useState<CapturedFile | null>(null);
  const [result, setResult] = useState<CombinedResult | null>(null);
  /** Incremented whenever we navigate away from input → tells InputScreen to
   *  tear down its mic / AudioContext / rAF. */
  const [teardownSignal, setTeardownSignal] = useState(0);

  const lastAnalyseMode = useRef<Mode>("voice");

  const leaveInput = useCallback(() => setTeardownSignal((n) => n + 1), []);

  const goHome = useCallback(() => {
    leaveInput();
    setScreen("home");
  }, [leaveInput]);

  const goAbout = useCallback(() => {
    leaveInput();
    setScreen("about");
  }, [leaveInput]);

  const startMode = useCallback(
    (m: Mode) => {
      leaveInput();
      setMode(m);
      setVoice(null);
      setMri(null);
      setResult(null);
      setScreen("input");
    },
    [leaveInput],
  );

  const startSelected = useCallback(() => {
    leaveInput();
    setResult(null);
    setScreen("input");
  }, [leaveInput]);

  const backToInput = useCallback(() => setScreen("input"), []);

  const runPredict = useCallback(
    async (m: Mode, v: CapturedFile | null, i: CapturedFile | null) => {
      lastAnalyseMode.current = m;
      leaveInput();
      setScreen("processing");
      const start = Date.now();
      try {
        const res = await predict(m, {
          voice: v?.blob ?? null,
          image: i?.blob ?? null,
        });
        const wait = Math.max(0, MIN_PROCESSING_MS - (Date.now() - start));
        window.setTimeout(() => {
          setResult(res);
          setScreen("results");
        }, wait);
      } catch {
        const wait = Math.max(0, MIN_PROCESSING_MS - (Date.now() - start));
        window.setTimeout(() => setScreen("error"), wait);
      }
    },
    [leaveInput],
  );

  const analyse = useCallback(() => {
    void runPredict(mode, voice, mri);
  }, [runPredict, mode, voice, mri]);

  const retry = useCallback(() => {
    void runPredict(lastAnalyseMode.current, voice, mri);
  }, [runPredict, voice, mri]);

  return (
    <div
      style={{
        minHeight: "100vh",
        background: "#F4F1EB",
        fontFamily: "'IBM Plex Sans',system-ui,sans-serif",
        color: "#15302E",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <TopBanner />
      <Header screen={screen} onHome={goHome} onAbout={goAbout} />

      <main
        style={{
          flex: 1,
          width: "100%",
          maxWidth: 1180,
          margin: "0 auto",
          padding: "14px 28px 72px",
        }}
      >
        {screen === "home" && (
          <Home
            onStart={startSelected}
            onAbout={goAbout}
            onVoice={() => startMode("voice")}
            onMri={() => startMode("mri")}
            onCombined={() => startMode("combined")}
          />
        )}

        {screen === "input" && (
          <InputScreen
            mode={mode}
            voice={voice}
            mri={mri}
            onMode={setMode}
            onVoice={setVoice}
            onMri={setMri}
            onAnalyse={analyse}
            onHome={goHome}
            teardownSignal={teardownSignal}
          />
        )}

        {screen === "processing" && <Processing mode={mode} />}

        {screen === "error" && (
          <ErrorScreen onRetry={retry} onBack={backToInput} />
        )}

        {screen === "results" && result && (
          <Results
            result={result}
            mode={mode}
            mriURL={mri?.url ?? null}
            onNewScreening={backToInput}
          />
        )}

        {screen === "about" && <About onHome={goHome} />}
      </main>

      <Footer onAbout={goAbout} />
    </div>
  );
}
