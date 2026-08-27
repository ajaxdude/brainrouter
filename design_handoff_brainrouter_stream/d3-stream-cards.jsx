// Direction 3: Stream Cards — chronological card feed; each task is self-contained card
const T3 = window.BR.TOKENS;
const { useState: useState3 } = React;

function StreamCards() {
  const [filter, setFilter] = useState3('all');
  const tasks = BR.TASKS.filter(t => filter === 'all' ? true : t.status === filter || (filter === 'live' && ['queued','generating','reviewing'].includes(t.status)));
  return (
    <div style={{ display: 'flex', height: '100%', background: T3.bg, color: T3.text, fontFamily: 'Inter, sans-serif' }}>
      <BRSidebar active="live" />
      <main style={{ flex: 1, display: 'flex', flexDirection: 'column', minWidth: 0, overflow: 'auto' }}>
        <BRCommandBar
          title="Stream"
          subtitle="Newest first · self-contained cards · review nested under each task"
          right={<div style={{ display: 'flex', gap: 6 }}>
            {['all','live','approved','escalated'].map(f => (
              <button key={f} className="br-btn" onClick={() => setFilter(f)} style={filter===f?{background:T3.bg3, color:T3.text}:{}}>{f}</button>
            ))}
          </div>}
        />

        <div style={{ padding: 24, display: 'grid', gridTemplateColumns: '1fr 320px', gap: 24, alignItems: 'start' }}>
          {/* Feed */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            {tasks.map((t, i) => <StreamCard key={t.id} task={t} isLast={i === tasks.length - 1} />)}
          </div>

          {/* Sticky right rail: aggregate sankey + stats */}
          <aside style={{ position: 'sticky', top: 0, display: 'flex', flexDirection: 'column', gap: 14 }}>
            <div style={{ background: T3.panel, border: `1px solid ${T3.border}`, borderRadius: 10, padding: 14 }}>
              <div style={{ fontSize: 10, letterSpacing: 1.5, color: T3.textMute, marginBottom: 8 }}>FLOW · 24h</div>
              <BRSankey width={300} height={240} />
            </div>
            <div style={{ background: T3.panel, border: `1px solid ${T3.border}`, borderRadius: 10, padding: 14, display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
              <Mini label="Total" value="225" c={T3.cyan} />
              <Mini label="Direct cloud" value="204" c={T3.green} sub="91%" />
              <Mini label="Cloud (auto)" value="11" c={T3.violet} sub="5%" />
              <Mini label="Local (auto)" value="6" c={T3.cyan} sub="3%" />
              <Mini label="Fallbacks" value="3" c={T3.amber} />
              <Mini label="Avg" value="3.4s" c={T3.cyan} />
            </div>
          </aside>
        </div>
      </main>
    </div>
  );
}

function Mini({ label, value, c, sub }) {
  return <div style={{ padding: 8, background: T3.bg2, borderRadius: 6, border: `1px solid ${T3.border}` }}>
    <div style={{ fontSize: 9, letterSpacing: 1.4, color: T3.textMute }}>{label.toUpperCase()}</div>
    <div style={{ fontSize: 18, fontFamily: 'JetBrains Mono, monospace', color: c, marginTop: 2 }}>{value}</div>
    {sub && <div style={{ fontSize: 10, color: T3.textMute }}>{sub}</div>}
  </div>;
}

function StreamCard({ task }) {
  const s = BR.SOURCES[task.source];
  const h = BR.HARNESSES[task.harness];
  const m = BR.MODELS[task.model];
  const r = task.review;
  const live = ['generating','reviewing'].includes(task.status);
  return (
    <article style={{
      background: T3.panel, border: `1px solid ${T3.border}`,
      borderLeft: `3px solid ${s.color}`,
      borderRadius: 10, overflow: 'hidden', position: 'relative',
    }}>
      {live && <div className="br-stream" style={{ position: 'absolute', top: 0, left: 0, right: 0, height: 2 }} />}
      {/* Header: source dot + meta + status */}
      <header style={{ padding: '12px 16px', display: 'flex', alignItems: 'center', gap: 10, borderBottom: `1px solid ${T3.border}` }}>
        <div style={{ width: 28, height: 28, borderRadius: 8, background: s.color + '22', border: `1px solid ${s.color}55`, display: 'flex', alignItems: 'center', justifyContent: 'center', color: s.color, fontFamily: 'JetBrains Mono, monospace', fontWeight: 600, fontSize: 13 }}>{s.glyph}</div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontSize: 12, color: T3.text }}>
            <span style={{ color: s.color, fontWeight: 600 }}>{s.label}</span>
            <span style={{ color: T3.textMute, marginLeft: 6 }}>{task.sourceMeta}</span>
          </div>
          <div style={{ fontSize: 10.5, color: T3.textMute, fontFamily: 'JetBrains Mono, monospace', marginTop: 2 }}>
            {task.taskId} · {task.folder} · {task.startedAt}
          </div>
        </div>
        <BRStatusPill status={task.status} />
      </header>

      {/* Flow rail */}
      <div style={{ padding: '10px 16px', background: T3.bg2, borderBottom: `1px solid ${T3.border}` }}>
        <BRMiniFlow task={task} />
      </div>

      {/* Body */}
      <div style={{ padding: 16, display: 'grid', gridTemplateColumns: '1fr 200px', gap: 16 }}>
        <div>
          <div style={{ fontSize: 13, color: T3.text, lineHeight: 1.55 }}>{task.prompt}</div>
          {task.streamingHint && <div style={{ fontSize: 11, color: T3.green, marginTop: 8, fontFamily: 'JetBrains Mono, monospace' }}>● {task.streamingHint}</div>}
          {(task.status === 'generating' || task.status === 'reviewing') && (
            <div style={{ marginTop: 10 }}>
              <BRProgress value={task.progress} color={task.status === 'reviewing' ? T3.amber : T3.green} height={3} />
            </div>
          )}
          {task.failReason && (
            <div style={{ marginTop: 10, padding: 10, background: T3.redDim, borderLeft: `2px solid ${T3.red}`, borderRadius: 4, fontSize: 12, color: T3.text }}>
              {task.failReason}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, padding: 10, background: T3.bg2, borderRadius: 6, fontSize: 11, fontFamily: 'JetBrains Mono, monospace' }}>
          <Stat2 k="route" v={task.route} c={task.route.startsWith('cloud') ? T3.violet : T3.cyan} />
          <Stat2 k="model" v={m.label} c={m.color} />
          <Stat2 k="latency" v={task.latencyMs ? (task.latencyMs/1000).toFixed(1)+'s' : '—'} c={task.latencyMs > 5000 ? T3.amber : T3.text} />
          <Stat2 k="tokens" v={`${task.tokens.in/1000|0}k → ${task.tokens.out/1000|0}k`} c={T3.textDim} />
        </div>
      </div>

      {/* Review subtask — nested */}
      {r && r.status !== 'pending' && r.status !== 'failed' && (
        <div style={{ borderTop: `1px solid ${T3.border}`, background: T3.bg2, padding: '12px 16px 14px', position: 'relative' }}>
          <div style={{ position: 'absolute', left: 30, top: -1, bottom: 'auto', width: 24, height: 14, borderLeft: `1px dashed ${T3.borderHi}`, borderBottom: `1px dashed ${T3.borderHi}`, borderBottomLeftRadius: 6 }} />
          <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginLeft: 36 }}>
            <span style={{ fontSize: 10, color: T3.textMute, fontFamily: 'JetBrains Mono, monospace', letterSpacing: 1 }}>↳ REVIEW</span>
            <BRStatusPill status={r.status} />
            <span style={{ fontSize: 11, color: T3.textMute, fontFamily: 'JetBrains Mono, monospace' }}>iter {r.iters}/5 · by {r.reviewer}</span>
            {r.status === 'escalated' && <span style={{ marginLeft: 'auto', fontSize: 10, color: T3.red, fontFamily: 'JetBrains Mono, monospace' }}>HUMAN NEEDED</span>}
          </div>
          {r.lastFeedback && (
            <div style={{ marginLeft: 36, marginTop: 8, padding: 10, background: r.status === 'escalated' ? T3.redDim : T3.amberDim, borderLeft: `2px solid ${r.status === 'escalated' ? T3.red : T3.amber}`, borderRadius: 4, fontSize: 12, color: T3.text, lineHeight: 1.5 }}>
              {r.lastFeedback}
            </div>
          )}
          {r.status === 'escalated' && (
            <div style={{ marginLeft: 36, marginTop: 10, display: 'flex', gap: 6 }}>
              <button className="br-btn" style={{ background: T3.redDim, color: T3.red, borderColor: T3.red+'55' }}>Send Feedback</button>
              <button className="br-btn" style={{ background: T3.amberDim, color: T3.amber, borderColor: T3.amber+'55' }}>Continue iterating</button>
              <button className="br-btn" style={{ background: T3.greenDim, color: T3.green, borderColor: T3.green+'55' }}>LGTM</button>
            </div>
          )}
        </div>
      )}
    </article>
  );
}

function Stat2({ k, v, c }) {
  return <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
    <span style={{ color: T3.textMute }}>{k}</span>
    <span style={{ color: c, textAlign: 'right', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{v}</span>
  </div>;
}

window.StreamCards = StreamCards;
