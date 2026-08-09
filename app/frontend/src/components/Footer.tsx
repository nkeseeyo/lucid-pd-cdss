interface Props {
  onAbout: () => void;
}

export default function Footer({ onAbout }: Props) {
  return (
    <footer style={{ borderTop: "1px solid #E4DDCE", background: "#EFEAE0" }}>
      <div
        style={{
          maxWidth: 1180,
          margin: "0 auto",
          padding: "20px 28px",
          display: "flex",
          flexWrap: "wrap",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 12,
        }}
      >
        <span style={{ fontSize: 12.5, color: "#7A8784" }}>
          LUCID-PD CDSS · Decision support, not diagnosis · Not a medical device
        </span>
        <button
          data-nv-focusring
          onClick={onAbout}
          style={{
            background: "none",
            border: "none",
            cursor: "pointer",
            font: "600 12.5px 'IBM Plex Sans'",
            color: "#0C5C5E",
            padding: 4,
          }}
        >
          About &amp; methodology
        </button>
      </div>
    </footer>
  );
}
