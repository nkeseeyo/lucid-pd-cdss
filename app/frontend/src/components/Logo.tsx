/** The LUCID-PD logo mark — "lens + pulse": a teal rounded square with a clear
 *  lens (lucidity) crossed by a clinical pulse line. `size` controls the box. */
export default function Logo({ size = 30 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 64 64"
      role="img"
      aria-label="LUCID-PD"
      style={{ flex: "none", display: "block" }}
    >
      <defs>
        <linearGradient id="lucid-mark" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#0C5C5E" />
          <stop offset="1" stopColor="#157E78" />
        </linearGradient>
      </defs>
      <rect width="64" height="64" rx="15" fill="url(#lucid-mark)" />
      <circle cx="32" cy="32" r="16" fill="none" stroke="#fff" strokeWidth="3.5" />
      <path
        d="M19 32h6l3-7 5 14 3-7h6"
        fill="none"
        stroke="#fff"
        strokeWidth="3.2"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}
