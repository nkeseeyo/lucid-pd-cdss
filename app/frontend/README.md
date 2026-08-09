# NeuroVox: the LUCID-PD interface

React with Vite and TypeScript. It talks to the FastAPI service described in
[../README.md](../README.md) and holds no model logic of its own.

## Running it

```bash
npm install
npm run dev          # development server, talks to http://127.0.0.1:8000
npm run build        # type-check, then emit the production bundle to dist/
```

`VITE_API_URL` overrides the backend origin; copy `.env.example` to `.env` to set it. With
no override, the development server calls the local FastAPI process and a production build
calls its own origin, because the deployed container serves this bundle and the API
together.

`npm run build` runs `tsc --noEmit` first, so a type error fails the build rather than
shipping.

## Structure

`App.tsx` holds the screen state machine: home, input, processing, results, error. Each
screen is a component under `src/components/`, `api.ts` is the only place that talks to the
backend, `types.ts` mirrors the Pydantic response models, and `theme.ts` with `styles.css`
carry the styling. There is no router, no state library and no component framework; the
only runtime dependencies are React and React DOM.

## Behaviour worth knowing

- Three modes, voice, MRI and combined, mirroring the API routes.
- Results show the risk band, the SHAP feature evidence or the Grad-CAM overlay, the
  grounded explanation and the care-route card.
- A "not a medical device, not a diagnosis" banner is present on every screen, and the MRI
  mode additionally carries its leakage caveat.
- A failed request routes to the error screen. There is no mock fallback, so the interface
  never displays a fabricated result when the backend is unreachable.
