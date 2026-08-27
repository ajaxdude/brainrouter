// Shared building blocks for all directions

const { useState, useMemo, useEffect, useRef } = React;
const T = window.BR.TOKENS;

// ─────────────────────────────────────────────────────────────────────────────
// Sidebar / Command Bar
// ─────────────────────────────────────────────────────────────────────────────
function BRSidebar({ active = 'live', compact = false }) {
  const items = [
    { k: 'live',    label: 'Live',          icon: '◉' },
    { k: 'history', label: 'History',       icon: '≡' },
    { k: 'reviews', label: 'Reviews',       icon: '✓' },
    { k: 'routing', label: 'Routing',       icon: '⇄' },
    { k: 'models',  label: 'Models',        icon: '◊' },
    { k: 'bridges', label: 'Bridges',       icon: '⌥' },
    { k: 'config',  label: 'Config',        icon: '⚙' },
  ];
  const tools = [
    { k: 'restart-stack',  label: 'Restart Local Stack' },
    { k: 'restart-mfst',   label: 'Restart Manifest' },
    { k: 'restart-br',     label: 'Restart brainrouter' },
    { k: 'sync-models',    label: 'Sync Models' },
  ];
  return (
    <aside style={{
      width: compact ? 64 : 224,
      background: T.bg2,
      borderRight: `1px solid ${T.border}`,
      display: 'flex', flexDirection: 'column',
      fontSize: 13, color: T.text,
      flexShrink: 0,
    }}>
      <div style={{ padding: compact ? '20px 0' : '20px 18px', borderBottom: `1px solid ${T.border}`, display: 'flex', alignItems: 'center', gap: 10, justifyContent: compact ? 'center' : 'flex-start' }}>
        <div style={{ width: 28, height: 28, borderRadius: 8, background: `linear-gradient(135deg, ${T.green}, ${T.cyan})`, display: 'flex', alignItems: 'center', justifyContent: 'center', fontWeight: 700, fontSize: 13, color: T.bg }}>br</div>
        {!compact && <div>
          <div style={{ fontWeight: 600, letterSpacing: 0.2 }}>brainrouter</div>
          <div style={{ fontSize: 10, color: T.textMute, fontFamily: 'JetBrains Mono, monospace' }}>v1.1.3</div>
        </div>}
      </div>

      <nav style={{ padding: '12px 8px', display: 'flex', flexDirection: 'column', gap: 2 }}>
        {items.map(it => (
          <button key={it.k} style={{
            all: 'unset', cursor: 'pointer',
            padding: compact ? '10px 0' : '8px 12px',
            borderRadius: 8,
            display: 'flex', alignItems: 'center', gap: 10, justifyContent: compact ? 'center' : 'flex-start',
            background: active === it.k ? T.bg3 : 'transparent',
            color: active === it.k ? T.text : T.textDim,
            borderLeft: active === it.k ? `2px solid ${T.green}` : '2px solid transparent',
            fontSize: 13,
          }}>
            <span style={{ fontSize: 14, width: 16, textAlign: 'center', color: active === it.k ? T.green : T.textMute }}>{it.icon}</span>
            {!compact && <span>{it.label}</span>}
            {!compact && it.k === 'reviews' && <span style={{ marginLeft: 'auto', fontSize: 10, padding: '2px 6px', borderRadius: 99, background: T.amberDim, color: T.amber, fontFamily: 'JetBrains Mono, monospace' }}>3</span>}
          </button>
        ))}
      </nav>

      {!compact && <>
        <div style={{ padding: '8px 18px', fontSize: 10, letterSpacing: 1.5, color: T.textMute, marginTop: 8 }}>SYSTEM HEALTH</div>
        <div style={{ padding: '0 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
          {[
            ['llama-swap', BR.HEALTH.llamaSwap.ver, true],
            ['llama.cpp',  BR.HEALTH.llamaCpp.ver,  true],
            ['toolbox',    BR.HEALTH.toolbox.ver,   true],
            ['manifest',   '9d2dad4',                true],
          ].map(([n,v,up]) => (
            <div key={n} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', fontSize: 12 }}>
              <span style={{ width: 6, height: 6, borderRadius: 99, background: up ? T.green : T.red, boxShadow: up ? `0 0 8px ${T.green}` : 'none' }} />
              <span style={{ color: T.textDim, flex: 1 }}>{n}</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', color: T.textMute, fontSize: 11 }}>{v}</span>
            </div>
          ))}
        </div>

        <div style={{ padding: '12px 18px 6px', fontSize: 10, letterSpacing: 1.5, color: T.textMute, marginTop: 12 }}>BRIDGES</div>
        <div style={{ padding: '0 12px', display: 'flex', flexDirection: 'column', gap: 4 }}>
          {[['discord', T.violet, '0 msg · 50h'], ['signal', T.cyan, '0 msg · 50h'], ['local', T.green, 'live']].map(([n,c,m]) => (
            <div key={n} style={{ display: 'flex', alignItems: 'center', gap: 8, padding: '6px 10px', fontSize: 12 }}>
              <span style={{ width: 6, height: 6, borderRadius: 99, background: c, boxShadow: `0 0 8px ${c}` }} />
              <span style={{ color: T.textDim, flex: 1, textTransform: 'capitalize' }}>{n}</span>
              <span style={{ fontFamily: 'JetBrains Mono, monospace', color: T.textMute, fontSize: 10 }}>{m}</span>
            </div>
          ))}
        </div>

        <div style={{ marginTop: 'auto', padding: 12, borderTop: `1px solid ${T.border}`, display: 'flex', flexDirection: 'column', gap: 4 }}>
          {tools.map(t => (
            <button key={t.k} style={{
              all: 'unset', cursor: 'pointer',
              padding: '7px 10px', borderRadius: 6,
              fontSize: 12, color: T.textDim,
              border: `1px solid ${T.border}`,
              textAlign: 'left',
            }}>↻ {t.label}</button>
          ))}
        </div>
      </>}
    </aside>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Top command bar (replaces top nav)
// ─────────────────────────────────────────────────────────────────────────────
function BRCommandBar({ title, subtitle, right }) {
  return (
    <div style={{ padding: '14px 24px', borderBottom: `1px solid ${T.border}`, background: T.bg2, display: 'flex', alignItems: 'center', gap: 16 }}>
      <div style={{ flex: 1 }}>
        <div style={{ fontSize: 16, fontWeight: 600, color: T.text }}>{title}</div>
        {subtitle && <div style={{ fontSize: 12, color: T.textMute, marginTop: 2 }}>{subtitle}</div>}
      </div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, background: T.bg3, padding: '6px 12px', borderRadius: 8, border: `1px solid ${T.border}`, minWidth: 320 }}>
        <span style={{ color: T.textMute, fontSize: 12 }}>⌕</span>
        <span style={{ color: T.textMute, fontSize: 12, flex: 1 }}>Search tasks, models, sources…</span>
        <span style={{ color: T.textMute, fontSize: 10, fontFamily: 'JetBrains Mono, monospace', padding: '2px 6px', border: `1px solid ${T.border}`, borderRadius: 4 }}>⌘K</span>
      </div>
      {right}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Badges
// ─────────────────────────────────────────────────────────────────────────────
function BRBadge({ color = T.green, children, dim = false, style = {} }) {
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '3px 8px', borderRadius: 5,
      fontSize: 10.5, fontFamily: 'JetBrains Mono, monospace',
      letterSpacing: 0.4, textTransform: 'uppercase',
      background: dim ? 'transparent' : color + '22',
      color: color,
      border: `1px solid ${color}55`,
      whiteSpace: 'nowrap',
      ...style,
    }}>{children}</span>
  );
}

function BRSourceBadge({ srcKey }) {
  const s = BR.SOURCES[srcKey];
  return <BRBadge color={s.color}>
    <span style={{ width: 5, height: 5, borderRadius: 99, background: s.color, boxShadow: `0 0 6px ${s.color}` }} />
    {s.label}
  </BRBadge>;
}

function BRStatusPill({ status }) {
  const map = {
    queued:     { c: T.textMute, l: 'Queued' },
    routing:    { c: T.cyan,     l: 'Routing' },
    generating: { c: T.green,    l: 'Generating', pulse: true },
    reviewing:  { c: T.amber,    l: 'Reviewing',  pulse: true },
    approved:   { c: T.green,    l: 'Approved' },
    escalated:  { c: T.red,      l: 'Escalated' },
    failed:     { c: T.red,      l: 'Failed' },
    iterating:  { c: T.amber,    l: 'Iterating' },
    pending:    { c: T.textMute, l: 'Pending' },
  };
  const x = map[status] || map.pending;
  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 6,
      padding: '3px 8px', borderRadius: 99, fontSize: 11,
      background: x.c + '20', color: x.c,
      border: `1px solid ${x.c}40`, whiteSpace: 'nowrap', fontFamily: 'JetBrains Mono, monospace', letterSpacing: 0.3,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: 99, background: x.c,
        boxShadow: `0 0 6px ${x.c}`,
        animation: x.pulse ? 'brPulse 1.4s ease-in-out infinite' : 'none',
      }} />
      {x.l}
    </span>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Mini flow (per-task pipeline glyph): source → harness → model → reviewer
// ─────────────────────────────────────────────────────────────────────────────
function BRMiniFlow({ task, compact = false }) {
  const s = BR.SOURCES[task.source];
  const h = BR.HARNESSES[task.harness];
  const m = BR.MODELS[task.model];
  const r = task.review?.reviewer && task.review.reviewer !== '—' ? BR.MODELS[task.review.reviewer] : null;
  const node = (label, color, sub) => (
    <div style={{ display: 'flex', alignItems: 'center', gap: 6, padding: compact ? '3px 7px' : '5px 9px', borderRadius: 6, background: color + '14', border: `1px solid ${color}40`, color }}>
      <span style={{ width: 5, height: 5, borderRadius: 99, background: color, boxShadow: `0 0 5px ${color}` }} />
      <span style={{ fontSize: compact ? 10 : 11, fontFamily: 'JetBrains Mono, monospace', letterSpacing: 0.3 }}>{label}</span>
      {sub && !compact && <span style={{ color: T.textMute, fontSize: 10 }}>{sub}</span>}
    </div>
  );
  const arrow = <span style={{ color: T.textMute, fontSize: 11 }}>→</span>;
  return (
    <div style={{ display: 'inline-flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
      {node(s.label, s.color)}
      {arrow}
      {node(h.label, h.color)}
      {arrow}
      {node(m.label, m.color, m.kind)}
      {r && <>{arrow}{node('rev: ' + r.label, T.amber)}</>}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Aggregate Sankey (svg)
// ─────────────────────────────────────────────────────────────────────────────
function BRSankey({ width = 1100, height = 260 }) {
  const cols = [
    { key: 'sources',   nodes: BR.SANKEY.sources,   color: (k) => BR.SOURCES[k].color,   label: (k) => BR.SOURCES[k].label,   header: 'SOURCE' },
    { key: 'harnesses', nodes: BR.SANKEY.harnesses, color: (k) => BR.HARNESSES[k].color, label: (k) => BR.HARNESSES[k].label, header: 'HARNESS' },
    { key: 'routes',    nodes: BR.SANKEY.routes,    color: (k) => k.startsWith('cloud') ? T.violet : T.cyan, label: (k) => k.toUpperCase(), header: 'ROUTE' },
    { key: 'models',    nodes: BR.SANKEY.models,    color: (k) => BR.MODELS[k].color,    label: (k) => BR.MODELS[k].label,    header: 'MODEL' },
  ];
  const colX = cols.map((_, i) => 40 + i * ((width - 80) / (cols.length - 1)));
  const nodeWidth = 14;
  const headerH = 22;
  const usableH = height - headerH - 20;
  // total per col
  const totals = cols.map(c => c.nodes.reduce((a,[,v]) => a + v, 0));
  const padPerCol = 6;
  // compute node y positions
  const colNodes = cols.map((c, ci) => {
    const total = totals[ci];
    const padTotal = (c.nodes.length - 1) * padPerCol;
    const scale = (usableH - padTotal) / total;
    let y = headerH + 10;
    return c.nodes.map(([k, v], idx) => {
      const h = v * scale;
      const node = { key: k, value: v, y, h, x: colX[ci], cumOut: 0, cumIn: 0, color: c.color(k), label: c.label(k) };
      y += h + padPerCol;
      return node;
    });
  });
  const allNodes = {};
  colNodes.forEach(arr => arr.forEach(n => { allNodes[n.key] = n; }));

  // build links
  const linksOut = BR.SANKEY.links.map(([from, to, v]) => {
    const a = allNodes[from], b = allNodes[to];
    if (!a || !b) return null;
    const ah = (v / a.value) * a.h;
    const bh = (v / b.value) * b.h;
    const ay = a.y + a.cumOut + ah / 2;
    const by = b.y + b.cumIn + bh / 2;
    a.cumOut += ah;
    b.cumIn += bh;
    return { x1: a.x + nodeWidth, y1: ay, x2: b.x, y2: by, w: Math.max(1, Math.min(ah, bh)), color: a.color };
  }).filter(Boolean);

  const path = (l) => {
    const cx1 = l.x1 + (l.x2 - l.x1) * 0.5;
    const cx2 = l.x2 - (l.x2 - l.x1) * 0.5;
    return `M ${l.x1} ${l.y1} C ${cx1} ${l.y1}, ${cx2} ${l.y2}, ${l.x2} ${l.y2}`;
  };

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ display: 'block' }}>
      <defs>
        <linearGradient id="brSankFade" x1="0" x2="1">
          <stop offset="0" stopOpacity="0.55" />
          <stop offset="1" stopOpacity="0.18" />
        </linearGradient>
      </defs>
      {cols.map((c, ci) => (
        <text key={c.key} x={colX[ci]} y={14} fill={T.textMute} fontSize="9" fontFamily="JetBrains Mono, monospace" letterSpacing="1.5">{c.header}</text>
      ))}
      {linksOut.map((l, i) => (
        <path key={i} d={path(l)} stroke={l.color} strokeWidth={l.w} fill="none" opacity={0.32} />
      ))}
      {colNodes.flat().map((n, i) => (
        <g key={i}>
          <rect x={n.x} y={n.y} width={nodeWidth} height={n.h} fill={n.color} rx="2" />
          <text x={n.x + nodeWidth + 8} y={n.y + n.h / 2 + 3.5} fill={T.text} fontSize="11" fontFamily="JetBrains Mono, monospace">{n.label}</text>
          <text x={n.x + nodeWidth + 8} y={n.y + n.h / 2 + 16} fill={T.textMute} fontSize="9" fontFamily="JetBrains Mono, monospace">{n.value}</text>
        </g>
      ))}
    </svg>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// Stat tile
// ─────────────────────────────────────────────────────────────────────────────
function BRStat({ label, value, color = T.text, sub }) {
  return (
    <div style={{
      padding: '12px 16px', background: T.panel, border: `1px solid ${T.border}`, borderRadius: 8,
      display: 'flex', flexDirection: 'column', gap: 6, minWidth: 0,
    }}>
      <div style={{ fontSize: 9.5, color: T.textMute, letterSpacing: 1.5, textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 22, fontFamily: 'JetBrains Mono, monospace', color, fontWeight: 500 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: T.textMute }}>{sub}</div>}
    </div>
  );
}

// Progress bar
function BRProgress({ value = 0, color = T.green, height = 4 }) {
  return (
    <div style={{ height, background: T.bg3, borderRadius: 99, overflow: 'hidden', position: 'relative' }}>
      <div style={{
        width: `${Math.round(value * 100)}%`, height: '100%',
        background: `linear-gradient(90deg, ${color}, ${color}cc)`,
        boxShadow: `0 0 8px ${color}88`,
        transition: 'width 0.4s ease',
      }} />
    </div>
  );
}

// global keyframes — inject once
if (!document.getElementById('br-keyframes')) {
  const s = document.createElement('style');
  s.id = 'br-keyframes';
  s.textContent = `
    @keyframes brPulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(0.85); } }
    @keyframes brShimmer { 0% { background-position: -200% 0; } 100% { background-position: 200% 0; } }
    @keyframes brDash { to { stroke-dashoffset: -20; } }
    .br-stream { background: linear-gradient(90deg, transparent, ${T.green}33, transparent); background-size: 200% 100%; animation: brShimmer 1.8s linear infinite; }
    .br-row:hover { background: ${T.bg3} !important; }
    .br-btn { all: unset; cursor: pointer; padding: 6px 12px; border-radius: 6px; font-size: 12px; border: 1px solid ${T.border}; color: ${T.textDim}; }
    .br-btn:hover { color: ${T.text}; border-color: ${T.borderHi}; background: ${T.bg3}; }
    body { background: ${T.bg}; color: ${T.text}; font-family: Inter, -apple-system, sans-serif; margin: 0; }
    * { box-sizing: border-box; }
  `;
  document.head.appendChild(s);
}

Object.assign(window, { BRSidebar, BRCommandBar, BRBadge, BRSourceBadge, BRStatusPill, BRMiniFlow, BRSankey, BRStat, BRProgress });
