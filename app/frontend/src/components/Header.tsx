import Logo from "./Logo";
import type { Screen } from "../types";

interface Props {
  screen: Screen;
  onHome: () => void;
  onAbout: () => void;
}

const hoverBg = (e: React.MouseEvent<HTMLButtonElement>, on: boolean) => {
  e.currentTarget.style.background = on ? "#E6F0EE" : "none";
};

export default function Header({ screen, onHome, onAbout }: Props) {
  return (
    <header
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 16,
        padding: "15px 28px",
        maxWidth: 1180,
        margin: "0 auto",
        width: "100%",
      }}
    >
      <button
        data-nv-focusring
        onClick={onHome}
        aria-label="LUCID-PD CDSS home"
        style={{
          display: "flex",
          alignItems: "center",
          gap: 11,
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: 4,
        }}
      >
        <Logo />
        <span
          style={{
            fontWeight: 700,
            fontSize: 19,
            letterSpacing: "-.01em",
            color: "#0C5C5E",
          }}
        >
          LUCID-PD{" "}
          <span
            style={{
              fontFamily: "'IBM Plex Mono',monospace",
              fontWeight: 500,
              fontSize: 12,
              color: "#5A6A68",
              letterSpacing: ".04em",
              verticalAlign: 2,
            }}
          >
            CDSS
          </span>
        </span>
      </button>
      <nav
        aria-label="Primary"
        style={{ display: "flex", alignItems: "center", gap: 4 }}
      >
        <button
          data-nv-focusring
          onClick={onHome}
          aria-current={screen === "home" ? "page" : "false"}
          onMouseEnter={(e) => hoverBg(e, true)}
          onMouseLeave={(e) => hoverBg(e, false)}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            font: "600 14.5px 'IBM Plex Sans'",
            color: "#3C4D4B",
            padding: "9px 14px",
            borderRadius: 8,
            minHeight: 40,
          }}
        >
          Home
        </button>
        <button
          data-nv-focusring
          onClick={onAbout}
          aria-current={screen === "about" ? "page" : "false"}
          onMouseEnter={(e) => hoverBg(e, true)}
          onMouseLeave={(e) => hoverBg(e, false)}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            font: "600 14.5px 'IBM Plex Sans'",
            color: "#3C4D4B",
            padding: "9px 14px",
            borderRadius: 8,
            minHeight: 40,
          }}
        >
          About &amp; methodology
        </button>
      </nav>
    </header>
  );
}
