#!/usr/bin/env node
'use strict';

// P5 Part A1/A4 (docs/design/p5-ops-redesign.md): per-role card pickers.
// Structural + behavioral checks for main_dashboard.html with Node's test
// runner and a synthetic DOM via `vm` (no browser, no server, no deps),
// following scripts/test-nav-ui.cjs.
//
// What must hold:
//   * three pickers, one per role — the shared text input is gone for good;
//   * the pin menu compiles the grouped /api/routing-models payload onto the
//     EXISTING assignment grammar (bare local key, cloud/<provider>/<id>), so
//     /api/roles/set is unchanged;
//   * per-card pinning round-trips: distinct cloud ids per role survive
//     load → save → load;
//   * an assignment the catalog cannot describe is preserved, never quietly
//     rewritten to `cloud`.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../src/escalation/templates/main_dashboard.html'), 'utf8');

const ROLES = ['main', 'review', 'subagent'];
// The retired shared input's id, built by concatenation so this file itself
// never reintroduces the token the grep proof forbids.
const LEGACY_INPUT = ['card', 'model'].join('-');

function ids() { return [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]); }
function requireOneId(id) { assert.equal(ids().filter(x => x === id).length, 1, `expected exactly one #${id}`); }
function extractFn(name) {
  let start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
  // Include a leading `async ` so the extracted text is valid standalone.
  if (html.slice(Math.max(0, start - 6), start) === 'async ') start -= 6;
  let depth = 0;
  for (let i = html.indexOf('{', start); i < html.length; i++) {
    if (html[i] === '{') depth++;
    if (html[i] === '}' && --depth === 0) return html.slice(start, i + 1);
  }
  throw new Error(`unterminated ${name}`);
}

// Minimal <select> stand-in with browser-ish option semantics: assigning a
// value that no rendered <option> carries clears the selection (that is what
// makes the round-trip below a real test — a wrongly compiled option value
// would silently drop the assignment instead of storing it).
class Select {
  constructor(id) { this.id = id; this._options = []; this._html = ''; this._value = ''; }
  set innerHTML(markup) {
    this._html = markup;
    this._options = [...markup.matchAll(/<option value="([^"]*)"/g)].map(m => m[1]);
    if (!this._options.includes(this._value)) this._value = this._options[0] || '';
  }
  get innerHTML() { return this._html; }
  get options() { return this._options; }
  set value(v) { this._value = this._options.includes(v) ? v : ''; }
  get value() { return this._value; }
}

function sandboxFor(functionNames, extra = {}) {
  const elements = new Map();
  for (const role of ROLES) elements.set(`card-${role}-model`, new Select(`card-${role}-model`));
  elements.set('role-scope', { value: 'global' });
  const sandbox = {
    document: { getElementById: id => elements.get(id) || null },
    CARD_ROLES: ROLES,
    routingCatalog: null,
    fetches: [],
    escHtml: s => String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'),
    async fetch(p, opts) {
      sandbox.fetches.push([p, opts]);
      return { ok: true, status: 200, json: async () => ({}) };
    },
    loadRoleCards: () => {},
    setRolesMessage: () => {},
    ...extra,
  };
  vm.createContext(sandbox);
  vm.runInContext(functionNames.map(extractFn).join('\n'), sandbox, { filename: 'role-cards-ui.js' });
  sandbox.__select = role => elements.get(`card-${role}-model`);
  return sandbox;
}

const CATALOG = {
  local: { models: ['gufo-qwen3.8-flash', 'ds4-flash'], error: null },
  cloud: { models: ['hf:deepseek/DeepSeek-V4'], error: null },
  cloud_entries: [
    { provider: 'synthetic', id: 'hf:deepseek/DeepSeek-V4' },
    { provider: 'deepinfra', id: 'deepseek-ai/DeepSeek-V4-Flash' },
  ],
  peers: [{ provider: 'fedora1', models: ['swift-1.5-qwen3.8-27b.mq4'], healthy: true, last_seen: '2026-11-04T10:00:00Z' }],
};

// 1. Structure: three pickers, no shared input/datalist anywhere.
test('each role owns one picker and the shared model input is gone', () => {
  for (const role of ROLES) requireOneId(`card-${role}-model`);
  assert.ok(!html.includes(LEGACY_INPUT), `the shared ${LEGACY_INPUT} input must stay deleted`);
  assert.ok(!html.includes('routing-models-list'), 'the shared picker datalist must stay deleted');
  assert.ok(html.includes(`for="card-main-model"`), 'picker labels must stay bound to their own picker');
  for (const role of ROLES) {
    assert.ok(html.includes(`id="card-${role}-model"`) && html.includes(`<option value="auto">auto</option>`),
      'every picker carries the baseline selector vocabulary');
  }
});

// 2. The pin menu is the assignment grammar (no new wire format).
test('pin entries compile to the existing assignment grammar', () => {
  const s = sandboxFor(['cardPinEntries']);
  const values = s.cardPinEntries(CATALOG).map(e => e.value);
  // `.join('|')` keeps the comparison inside one realm (the entries come back
  // from the vm context, whose Array prototype differs from ours).
  assert.equal(values.join('|'), [
    'gufo-qwen3.8-flash',                       // bare local key
    'ds4-flash',
    'cloud/synthetic/hf:deepseek/DeepSeek-V4',  // cloud/<provider>/<id>
    'cloud/deepinfra/deepseek-ai/DeepSeek-V4-Flash',
    'cloud/fedora1/swift-1.5-qwen3.8-27b.mq4',  // peers use the same qualified form
  ].join('|'));
  // A single cloud provider uses the unqualified `cloud/<id>` form — the exact
  // shape existing cards store (e.g. `cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017`).
  const single = { local: { models: [] }, cloud_entries: [{ provider: 'synthetic', id: 'hf:deepseek-ai/DeepSeek-V4.1-Flash-1017' }], peers: [] };
  assert.equal(s.cardPinEntries(single)[0].value, 'cloud/hf:deepseek-ai/DeepSeek-V4.1-Flash-1017');
  // Duplicates between registry and live discovery collapse to one option.
  const dup = { local: { models: ['a', 'a'] }, cloud_entries: [{ provider: 'p', id: 'a' }, { provider: 'p', id: 'a' }], peers: [] };
  assert.equal(s.cardPinEntries(dup).length, 2);
  assert.equal(s.cardPinEntries(null).length, 0);
});

// 3. Round-trip: distinct per-role pins survive load → save.
test('per-card pinning round-trips with distinct cloud ids per role', async () => {
  const assignments = {
    main: 'cloud/synthetic/hf:deepseek/DeepSeek-V4',
    review: 'cloud/fedora1/swift-1.5-qwen3.8-27b.mq4',
    subagent: 'gufo-qwen3.8-flash',
  };
  const s = sandboxFor(['currentScope', 'routingFetch', 'cardSelect', 'cardPinEntries', 'fillCardSelect', 'cardAssignment', 'saveRoleCards']);
  s.routingCatalog = CATALOG;
  for (const role of ROLES) s.fillCardSelect(s.__select(role), assignments[role]);
  for (const role of ROLES) {
    const sel = s.__select(role);
    assert.ok(sel.innerHTML.includes('Pin exactly this model'), 'pin optgroup missing');
    assert.equal(s.cardAssignment(role), assignments[role], `${role} picker lost its pin`);
  }
  assert.equal(new Set(ROLES.map(r => s.cardAssignment(r))).size, 3, 'the three cards must hold distinct pins');

  await s.saveRoleCards();
  const posts = s.fetches.filter(([p]) => p === '/api/roles/set').map(([, opts]) => JSON.parse(opts.body));
  assert.deepEqual(posts.map(p => p.role), ROLES);
  for (const post of posts) {
    assert.equal(post.scope, 'global');
    assert.equal(post.assignment, assignments[post.role], 'save must post the pinned grammar verbatim');
  }
  assert.equal(new Set(posts.map(p => p.assignment)).size, 3, 'distinct per-role cloud ids must be savable');
});

// 4. Baselines stay baselines; unknown assignments are preserved, not coerced.
test('unknown or baseline assignments map back without being rewritten', () => {
  const s = sandboxFor(['cardSelect', 'cardPinEntries', 'fillCardSelect', 'cardAssignment']);
  s.routingCatalog = CATALOG;
  for (const baseline of ['auto', 'local', 'cloud']) {
    s.fillCardSelect(s.__select('main'), baseline);
    assert.equal(s.cardAssignment('main'), baseline);
  }
  // A pin the catalog does not know (provider removed, model renamed) must
  // survive a reload instead of collapsing to `cloud`.
  const legacy = 'cloud/retired-vendor/legacy-model';
  s.fillCardSelect(s.__select('review'), legacy);
  assert.equal(s.cardAssignment('review'), legacy);
  assert.ok(s.__select('review').innerHTML.includes('Pinned (unknown model)'), 'unknown pin must be labelled');
});

// 5. A catalog refresh must not clobber an unsaved selection.
test('catalog refresh preserves the current selection', () => {
  const s = sandboxFor(['cardSelect', 'cardPinEntries', 'fillCardSelect', 'renderCardSelects', 'cardAssignment']);
  s.routingCatalog = CATALOG;
  s.fillCardSelect(s.__select('subagent'), 'ds4-flash');
  s.routingCatalog = Object.assign({}, CATALOG, { local: { models: ['ds4-flash', 'new-model'] } });
  s.renderCardSelects();
  assert.equal(s.cardAssignment('subagent'), 'ds4-flash');
  assert.ok(s.__select('subagent').innerHTML.includes('new-model'), 'refreshed catalog must reach the picker');
});
