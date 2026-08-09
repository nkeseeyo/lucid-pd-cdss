export default function TopBanner() {
  return (
    <div
      role="note"
      aria-label="Important: this is a decision-support and research tool, not a diagnosis"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        justifyContent: "center",
        background: "#E8F0EE",
        color: "#0B4B4D",
        borderBottom: "1px solid #D2E2DE",
        padding: "9px 18px",
        fontSize: 13.5,
        fontWeight: 500,
        textAlign: "center",
        lineHeight: 1.4,
      }}
    >
      <svg
        width="17"
        height="17"
        viewBox="0 0 24 24"
        fill="none"
        aria-hidden="true"
        style={{ flex: "none" }}
      >
        <circle cx="12" cy="12" r="9" stroke="#0B4B4D" strokeWidth="2" />
        <path d="M12 11v5" stroke="#0B4B4D" strokeWidth="2" strokeLinecap="round" />
        <circle cx="12" cy="7.6" r="1.2" fill="#0B4B4D" />
      </svg>
      <span>
        Decision support and research tool. <strong>Not a diagnosis</strong>, not a
        medical device. Always apply clinical judgement.
      </span>
    </div>
  );
}
