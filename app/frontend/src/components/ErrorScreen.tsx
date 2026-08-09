interface Props {
  onRetry: () => void;
  onBack: () => void;
}

export default function ErrorScreen({ onRetry, onBack }: Props) {
  return (
    <section
      role="alert"
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
      <span
        aria-hidden="true"
        style={{
          width: 64,
          height: 64,
          borderRadius: "50%",
          background: "#F8E6DF",
          display: "grid",
          placeItems: "center",
          marginBottom: 22,
        }}
      >
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none">
          <circle cx="12" cy="12" r="9" stroke="#A6361A" strokeWidth="2" />
          <path d="M12 7v6" stroke="#A6361A" strokeWidth="2" strokeLinecap="round" />
          <circle cx="12" cy="16.4" r="1.1" fill="#A6361A" />
        </svg>
      </span>
      <h1 style={{ fontSize: 23, fontWeight: 700, margin: "0 0 8px" }}>
        We couldn&rsquo;t complete the analysis
      </h1>
      <p
        style={{
          fontSize: 15,
          color: "#5A6A68",
          margin: "0 0 24px",
          maxWidth: "38ch",
        }}
      >
        The prediction service didn&rsquo;t respond. Your input is still here — please
        try again.
      </p>
      <div style={{ display: "flex", gap: 12 }}>
        <button
          data-nv-focusring
          onClick={onRetry}
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: 8,
            background: "#0C5C5E",
            color: "#fff",
            border: "none",
            borderRadius: 11,
            height: 48,
            padding: "0 22px",
            font: "600 15px 'IBM Plex Sans'",
            cursor: "pointer",
          }}
        >
          Try again
        </button>
        <button
          data-nv-focusring
          onClick={onBack}
          style={{
            background: "#fff",
            color: "#0C5C5E",
            border: "1.5px solid #CADEDB",
            borderRadius: 11,
            height: 48,
            padding: "0 20px",
            font: "600 15px 'IBM Plex Sans'",
            cursor: "pointer",
          }}
        >
          Back to input
        </button>
      </div>
    </section>
  );
}
