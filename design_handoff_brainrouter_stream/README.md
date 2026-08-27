# Handoff: brainrouter — Stream Cards Dashboard

## Overview
A redesign of the brainrouter dashboard as a single unified webapp view. Replaces the previous "separate tables for everything" layout with a chronological **stream of task cards**, each card showing the full intake → harness → model → review journey, with the review nested directly under the task as a sub-task (no more separate Review Sessions table).

The view also rolls up: a left command-bar sidebar (replacing the top nav, with system health + bridges + restart actions), a sticky right rail with an aggregate Sankey of routing flow + KPI tiles, and inline status/progress on every card.

## About the Design Files
The files in this bundle are **design references created in HTML/React** — prototypes showing the intended look and behavior, not production code to ship as-is. The task is to **recreate this design in brainrouter's existing webapp codebase** (whatever framework is in use — likely React/Next or the existing dashboard stack), wiring real data in place of the mocks in `data.jsx` and reusing the codebase's existing component primitives and styling conventions where they exist.

If the existing dashboard is server-rendered HTML/templates, port the structure faithfully but use the codebase's existing template engine, classes, and conventions rather than introducing React just for this.

## Fidelity
**High-fidelity.** Pixel values, colors, fonts, spacing, and interaction states are all final and intentional. Recreate to match.

## Files in this bundle
- `brainrouter redesign.html` — full canvas with all 4 explored directions for context
- `Stream Cards Standalone.html` — the chosen direction, isolated full-screen
- `data.jsx` — mock data shape for tasks, sources, harnesses, models, sankey, health
- `components.jsx` — shared building blocks (Sidebar, CommandBar, Badges, StatusPill, MiniFlow, Sankey, Stat, Progress)
- `d3-stream-cards.jsx` — the Stream Cards view itself
- `design-canvas.jsx` — only used by the multi-direction canvas; not needed for the standalone

Run `Stream Cards Standalone.html` in a browser to see the chosen direction in isolation.

---

## Layout

The view is a 2-column shell inside the app frame:

```
┌─────────────┬─────────────────────────────────────────────────────────┐
│  Sidebar    │  Command Bar (title · subtitle · search · filters)      │
│  (224px)    ├─────────────────────────────────────────────────────────┤
│             │                                          ┌─────────────┐│
│  · Logo     │  ┌──────────────────────────────────┐    │  FLOW (24h) ││
│  · Nav      │  │  Stream card 1                   │    │   Sankey    ││
│  · Health   │  │  ┌─ Review subtask (nested) ─┐   │    │   svg       ││
│  · Bridges  │  │  └──────────────────────────┘    │    │             ││
│  · Tools    │  └──────────────────────────────────┘    │  KPI tiles  ││
│             │  ┌──────────────────────────────────┐    │  2×3 grid   ││
│             │  │  Stream card 2 ...               │    │             ││
│             │  └──────────────────────────────────┘    │  (sticky)   ││
│             │                                          └─────────────┘│
└─────────────┴─────────────────────────────────────────────────────────┘
```

- **Page padding:** 24px around the main grid
- **Grid columns:** `1fr 320px` with `gap: 24px`, `align-items: start`
- **Card list:** flex column, `gap: 14px`
- **Right rail:** `position: sticky; top: 0;` flex column `gap: 14px`

---

## Design Tokens

All values are in `data.jsx` under `BR_TOKENS`. Use these as CSS variables in production.

### Colors
| Token        | Hex                       | Usage |
|--------------|---------------------------|-------|
| `bg`         | `#0b0e13`                 | App background |
| `bg2`        | `#11151c`                 | Sidebar / right-rail panels / card body strips |
| `bg3`        | `#161b24`                 | Hover, active row, search input |
| `panel`      | `#141923`                 | Card background |
| `border`     | `#1f2632`                 | Default 1px borders |
| `borderHi`   | `#2a3343`                 | Hover/elevated borders, dashed connector |
| `text`       | `#e6edf6`                 | Primary text |
| `textDim`    | `#8693a8`                 | Secondary text |
| `textMute`   | `#5a6679`                 | Tertiary / labels / metadata |
| `green`      | `#22d39a`                 | Local source · approved · generating · OK latency |
| `cyan`       | `#3ec5ff`                 | Signal source · stats · local route |
| `violet`     | `#9d7bff`                 | Discord source · cloud route · Anthropic models |
| `amber`      | `#f5b14a`                 | Reviewing · warnings · OpenCode · slow latency (>5s) |
| `red`        | `#ff6b6b`                 | Failed · escalated · destructive |
| `pink`       | `#ff7ac6`                 | Vibe harness |

Each accent has a `*Dim` variant — same hue at ~14–16% alpha — used for filled chip/pill backgrounds. E.g. `greenDim: rgba(34,211,154,0.14)`.

### Typography
- **UI font:** `Inter, -apple-system, sans-serif` — weights 400/500/600/700
- **Mono font:** `JetBrains Mono, monospace` — used for IDs, paths, model names, latency numbers, badges, all data
- **Sizes:** 9–10px (eyebrow labels, letter-spacing 1.4–1.5), 11px (metadata), 12–13px (body), 15–16px (titles), 18–22px (KPI numbers)

### Spacing
- Card padding: `16px`
- Card header padding: `12px 16px`
- Tile padding: `12px 16px`
- Inter-card gap: `14px`
- Page gutter: `24px`

### Radius
- Cards / panels: `10px`
- Tiles, buttons, inputs: `6–8px`
- Pills / dots: `99px`

### Shadows / glows
No drop-shadows. Glows are produced via `box-shadow: 0 0 6–8px <color>` on small status dots, and SVG `<feGaussianBlur stdDeviation=3>` for the wires/dots in the constellation (not used in Stream Cards).

---

## Components

### 1. Sidebar (`BRSidebar`, `components.jsx`)
- Width 224px, background `bg2`, right border `1px solid border`
- **Logo block:** 28px gradient square (`linear-gradient(135deg, green, cyan)`) with white "br" text + product name + version (mono)
- **Nav items:** icon glyph + label, active = `bg3` background + 2px left border in `green`. The "Reviews" item shows a count chip on the right (`amberDim` bg, `amber` text)
- **System Health section:** rows with a status dot (6px, glowing if up), service name, version (mono right-aligned)
- **Bridges section:** same row pattern, color-coded dot per source
- **Tools (bottom):** outlined buttons with `↻` glyph for Restart Local Stack / Manifest / brainrouter / Sync Models

### 2. Command Bar (`BRCommandBar`)
- Top of main content, padding `14px 24px`, `bg2` background, bottom border
- Left: title (16px/600) + subtitle (12px, `textMute`)
- Center: search input (`bg3`, rounded 8px, with `⌕` glyph and `⌘K` kbd hint, min-width 320px)
- Right: filter pills (`all` / `live` / `approved` / `escalated`) — active = `bg3` bg, dim text by default

### 3. Stream Card (`StreamCard`)
The hero component. Four stacked sections:

**a) Spine:** 3px left border in the source color (`s.color`). 1px border + `panel` bg, radius 10px.

**b) Live shimmer:** if status is `generating` or `reviewing`, a 2px tall gradient bar at the very top runs a left→right shimmer animation (`brShimmer 1.8s linear infinite`).

**c) Header** (`12px 16px`, bottom border):
- 28px square avatar in `s.color + '22'` bg with `s.color` border, mono glyph (D/S/L) inside
- Source label (bold, in source color) + `sourceMeta` (e.g. `@adrien · #brainrouter`) in `textMute`
- Below: `taskId · folder · startedAt` — all mono, 10.5px, `textMute`
- Right: `BRStatusPill`

**d) Flow rail** (`10px 16px`, `bg2`, bottom border):
- Renders `BRMiniFlow` — a chain of pill nodes (`source → harness → model → reviewer`) separated by `→` arrows
- Each pill: colored dot (5px, glowing) + mono label (11px) + small kind tag (`cloud`/`local`)

**e) Body** (`16px`, 2-column grid `1fr 200px`):
- Left: prompt text (13px, line-height 1.55), then optional streaming hint line (mono, 11px, `green`, prefixed with `●`), then optional progress bar
- Right: stats panel — `bg2` bg, padding 10px, radius 6px, mono 11px. 4 rows of `key — value` (route, model, latency, tokens). Latency text turns `amber` if >5000ms.

**f) Review subtask** (only if review exists and status is not `pending`/`failed`):
- Top border, `bg2` bg, padding `12px 16px 14px`
- An L-shaped dashed connector `1px dashed borderHi` in the top-left (24×14, borderLeft + borderBottom + radius 6 on bottom-left) — visually anchors the review to its parent
- Header row (indented 36px): `↳ REVIEW` mono eyebrow + status pill + `iter N/5 · by <reviewer>` + (if escalated) right-aligned `HUMAN NEEDED` in red
- Feedback box: `redDim` bg if escalated else `amberDim`, 2px left border in matching color, radius 4px, padding 10px, body text 12px
- Action row (escalated only): three buttons — Send Feedback (red), Continue iterating (amber), LGTM (green). Each uses the `*Dim` background + accent text + `accent55` border, 6×12 padding, radius 6.

### 4. Status Pill (`BRStatusPill`)
Small rounded-99 capsule, mono uppercase 11px, with a 6px dot. Dot pulses (`brPulse 1.4s`) for `generating` and `reviewing`. Color map: queued=mute, routing=cyan, generating=green, reviewing=amber, approved=green, escalated=red, failed=red, iterating=amber.

### 5. Mini Flow (`BRMiniFlow`)
Inline-flex row of node pills + `→` arrows. Each node = `5×9px` padding, color-tinted bg (`color + '14'`), bordered (`color + '40'`), mono label.

### 6. Sankey (`BRSankey`)
SVG, viewBox `0 0 width height`, `preserveAspectRatio` default. Four columns (Source, Harness, Route, Model) with column header text at top. Each column is a stack of colored `<rect>` nodes (14px wide), padded 6px between nodes, height = `value * scale`. Links are smooth horizontal Bézier curves (cubic) with stroke-width = link value, opacity 0.32, colored by source-side node. Right of each node: label (mono, 11px) + count below (mono, 9px, `textMute`).

### 7. Right Rail
- **Sankey panel:** `panel` bg, padding 14px, radius 10px. Eyebrow `FLOW · 24h` (10px, `textMute`, letter-spacing 1.5), then a 300×240 sankey
- **KPI grid:** 2×3 grid of mini tiles. Each tile: `bg2` bg, 6px radius, 1px border, padding 8px. Eyebrow uppercase 9px + big mono number (18px) in accent color + optional small `sub` text.

---

## Interactions

- **Filter pills** (top right of command bar): switch `filter` state. `'all'` = show all tasks, `'live'` = `queued|generating|reviewing`, `'approved'` and `'escalated'` filter to that status.
- **Status dot pulse:** CSS keyframe `brPulse` — opacity 1↔0.4 + scale 1↔0.85, 1.4s ease-in-out infinite. Applied only to live (`generating`/`reviewing`) statuses.
- **Live shimmer bar:** CSS keyframe `brShimmer` — `background-position: -200% 0 → 200% 0`, 1.8s linear infinite, with a horizontal gradient `transparent → green33 → transparent`.
- **Polling:** the dashboard should refetch every 3s (matches the original behavior). The "polling every 3s" copy in the subtitle reflects this. Use SWR/React Query/equivalent.
- **Card click:** in the prototype, cards aren't clickable. In production, clicking a card should open a detail/drawer view (or route to `/tasks/:id`) with the full prompt, complete event log, and review history.
- **Review action buttons:** wire to existing endpoints — `Send Feedback` posts the textarea content + status `feedback`; `Continue iterating` triggers another review pass; `LGTM` marks approved.
- **Search (`⌘K`):** open a command palette over tasks, models, sources, sessions.

---

## State Management

Per task in the feed:
- `id`, `taskId`, `folder`, `source`, `sourceMeta`, `harness`, `route`, `model`, `prompt`, `status`, `progress` (0–1), `startedAt`, `latencyMs`, `tokens: { in, out }`, `streamingHint?`, `failReason?`
- `review: { reviewer, status, iters, lastFeedback?, summary? }`

Global filters: `filter` (`all|live|approved|escalated`).

System health (sidebar): poll every ~10s — `llamaSwap`, `llamaCpp`, `toolbox`, `manifest` versions + up/down + latency; `bridges.discord` and `bridges.signal` connection state + msg count + uptime hours.

Sankey aggregate: query last-24h routing events bucketed by `source × harness × route × model`. Shape exactly the `BR_SANKEY` structure in `data.jsx`:
```js
{ sources: [[key, count], ...], harnesses: [...], routes: [...], models: [...],
  links: [[fromKey, toKey, count], ...] }
```

KPI tiles: `total`, `directCloud`, `cloudAuto`, `localAuto`, `fallbacks`, `avgLatency` — also from the last-24h window.

---

## Data shape reference (mock → real)

The mock keys map 1:1 to the existing brainrouter event feed:

| Mock field        | Source in current dashboard |
|-------------------|-----------------------------|
| `source`          | derived from bridge that received the prompt (discord/signal/local) |
| `sourceMeta`      | bridge-specific identifier (Discord username + channel, Signal number, local CWD) |
| `harness`         | which agent harness handled it (omp/cc/droid/vibe/opencode) |
| `route`           | the existing `ROUTE MODE` value (`cloud-direct`, `cloud-auto`, `local-auto`, `local-direct`) |
| `model`           | the existing `MODEL` value |
| `status`          | derive from event-feed `STATUS` + review session state |
| `progress`        | streaming token count / expected (or 1 if complete) |
| `latencyMs`       | the existing `LATENCY` field |
| `review.*`        | join from the Review Sessions table by `taskId` |

The big change: stop rendering Review Sessions as a separate table. Instead, when fetching tasks, left-join the review session and inline it on the card.

---

## Sources / Harnesses / Models reference

**Sources:** `discord` (violet), `signal` (cyan), `local` (green) — confirmed by user, no others.

**Harnesses:** `omp` (default, green), `droid` (cyan), `cc` Claude Code (violet), `opencode` (amber), `vibe` (pink).

**Models** (extend as needed in `MODELS`):
- `claude-sonnet-4-6` — cloud, anthropic, violet
- `claude-opus-4-6` — cloud, anthropic, violet
- `qwen3.6-27b-heretic` — local, llama-swap, cyan
- `gpt-5.2-mini` — cloud, openai, green
- `glm-4.6-air` — local, llama-swap, amber

---

## Implementation notes

- **Don't reach for animation libraries** for the pulse/shimmer — plain CSS keyframes (already defined in `components.jsx`) are sufficient.
- **The Sankey is hand-built SVG** — it doesn't need d3-sankey. The layout algo is in `BRSankey` and is ~30 lines. If you'd rather, swap to `d3-sankey` for production but match the visual styling (cubic Bézier links, opacity 0.32, source-colored, 14px node rects).
- **Use CSS variables** for the token palette so dark/light themes can swap later if needed.
- **Keep all data values monospaced.** The prototype is consistent: any number, ID, model name, version, or path is mono. UI labels and prose are Inter.
- **Reviews are the headline change.** The biggest UX delta from the old dashboard is that reviews are no longer a peer table — they're a child of their task. Don't be tempted to keep both.

## Assets
None required. All visuals are CSS/SVG. Icon glyphs are Unicode (`◉ ≡ ✓ ⇄ ◊ ⌥ ⚙ ↻ ⌕ ↳ ●`).
