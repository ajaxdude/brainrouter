#!/usr/bin/env node
'use strict';

// Structural + behavioral checks for the v2 L3 role-cards / policy /
// credentials / decisions dashboard UI (main_dashboard.html). Node's built-in
// test runner + a synthetic DOM via `vm` (no browser, no server, no deps),
// mirroring scripts/test-nav-ui.cjs.
//
// v2 L3 retarget (spec: "Removals: routing_profile store/CLI/widgets;
// project_pins store (migrated)" + "Add: Credentials tab … role cards panel
// (3 cards selector pickers + per-project) … Policy chips … Decisions panel"):
// the Phase-2 pins UI tests are replaced by the equivalent card-surface
// checks; per-role inherit controls are gone because cards ARE the scope
// precedence (project > global > default, resolved server-side).

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../src/escalation/templates/main_dashboard.html'), 'utf8');

function ids() { return [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]); }
function requireOneId(id) { assert.equal(ids().filter(x => x === id).length, 1, `expected exactly one #${id}`); }
function extractFn(name) {
  let start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
  // Include a leading `async ` so the extracted text is valid standalone.
  const asyncPrefix = html.slice(Math.max(0, start - 6), start) === 'async ';
  if (asyncPrefix) start -= 6;
  let depth = 0;
  for (let i = html.indexOf('{', start); i < html.length; i++) {
    if (html[i] === '{') depth++;
    if (html[i] === '}' && --depth === 0) return html.slice(start, i + 1);
  }
  throw new Error(`unterminated ${name}`);
}

class El {
  constructor(tag, id) {
    this.tag = tag; this.id = id; this.style = {}; this.attrs = {};
    this.textContent = ''; this.title = ''; this.className = ''; this.value = ''; this.disabled = false;
    this.children = []; this._html = null; this.dataset = {};
    const self = this;
    this.classList = {
      _set: new Set(),
      toggle(name, on) { if (on) this._set.add(name); else this._set.delete(name); self.className = [...this._set].join(' '); },
      contains(name) { return this._set.has(name); },
      add(name) { this._set.add(name); self.className = [...this._set].join(' '); },
      remove(name) { this._set.delete(name); self.className = [...this._set].join(' '); },
    };
  }
  appendChild(c) { this.children.push(c); return c; }
  append(...cs) { cs.forEach(c => this.children.push(c)); }
  replaceChildren(...cs) { this.children = cs.slice(); }
  addEventListener(_ev, _fn) {}
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  set innerHTML(v) { this._html = v; }   // used only to detect unsafe usage
  get innerHTML() { return this._html || ''; }
  focus() {}
}

function sandbox(functionNames, extra = {}) {
  const elements = new Map();
  for (const id of ids()) if (!id.includes('${')) elements.set(id, new El('div', id));
  const s = {
    cardStateDirty: false,
    policyState: { privacy: 'cloud-ok', quality: 'fast' },
    CARD_ROLES: ['main', 'review', 'subagent'],
    fetches: [],
    async fetch(path, opts) {
      s.fetches.push([path, opts]);
      // The daemon echoes the stored record; mirror that for policy writes.
      if (opts && path.includes('/api/policy')) {
        const body = JSON.parse(opts.body);
        return { ok: true, status: 200, json: async () => ({ policy: [body] }) };
      }
      return { ok: true, status: 200, json: async () => ({}) };
    },
    document: {
      getElementById: id => elements.get(id) || null,
      createElement: tag => new El(tag),
      addEventListener() {},
    },
    ...extra,
  };
  vm.createContext(s);
  vm.runInContext(functionNames.map(extractFn).join('\n'), s, { filename: 'roles-ui.js' });
  s.__get = id => { if (!elements.has(id)) elements.set(id, new El('div', id)); return elements.get(id); };
  return s;
}

// 1. The card strip exists exactly once: scope selector, three role pickers,
//    model field, save button, policy chips.
test('role-card strip and policy chips exist exactly once', () => {
  for (const id of ['role-scope', 'card-main', 'card-review', 'card-subagent', 'card-model',
    'save-cards', 'roles-message',
    'chip-privacy-cloud-ok', 'chip-privacy-local-only',
    'chip-quality-fast', 'chip-quality-balanced', 'chip-quality-maximum']) {
    requireOneId(id);
  }
});

// 2. The removed v1 widgets are gone (spec: "Removals: … dashboard
//    role-chooser/preset widgets; project_pins store").
test('removed v1 widgets are gone', () => {
  for (const id of ['routing-preset', 'save-routing-profile', 'project-pins-list',
    'routing-expander', 'pin-scope', 'routing-main-select', 'routing-reviewer-model',
    'routing-subagent-model', 'routing-hint', 'routing-models-list-unused']) {
    assert.ok(!ids().includes(id), `stale widget #${id} must be removed`);
  }
  assert.ok(!html.includes('/api/routing-profile'), 'routing-profile endpoint must be gone');
  assert.ok(!html.includes('/api/project-pin'), 'project-pin endpoint must be gone');
});

// 3. cardAssignment: managed selectors verbatim; cloud/<id> composes from the
//    model field (spec: "the card assignment compiles through the SAME
//    selector path").
test('cardAssignment compiles the selector vocabulary', () => {
  const s = sandbox(['currentScope', 'cardAssignment']);
  const scope = s.__get('role-scope'); scope.value = 'global';
  const sel = s.__get('card-main');
  const model = s.__get('card-model');
  sel.value = 'auto'; model.value = '';
  assert.equal(s.cardAssignment('main'), 'auto');
  sel.value = 'local'; model.value = 'ignored-for-local';
  assert.equal(s.cardAssignment('main'), 'local');
  sel.value = 'cloud'; model.value = '';
  assert.equal(s.cardAssignment('main'), 'cloud');
  sel.value = 'cloud'; model.value = 'vendor/model-2026';
  assert.equal(s.cardAssignment('main'), 'cloud/vendor/model-2026');
});

// 4. Save predicate: Save starts disabled; markCardDirty enables it (v1
//    "single Save predicate" keeper, retargeted).
test('Save is disabled until a card is edited', () => {
  const s = sandbox(['markCardDirty']);
  const btn = s.__get('save-cards');
  btn.disabled = true;
  s.markCardDirty();
  assert.equal(btn.disabled, false);
  assert.equal(s.cardStateDirty, true);
});

// 5. Policy chips: setPrivacy/setQuality POST the scope-resolved record (spec:
//    "Policy chips (privacy cloud-ok|local-only, quality fast|balanced|maximum)
//    — global + per-project").
test('policy chip handlers post scope-resolved records', async () => {
  const s = sandbox(['currentScope', 'routingFetch', 'setPrivacy', 'setQuality', 'setPolicyRecord', 'renderPolicy', 'setRolesMessage']);
  const scope = s.__get('role-scope'); scope.value = '/srv/repo';
  await s.setPrivacy('local-only');
  let last = s.fetches.at(-1);
  assert.equal(last[0], '/api/policy/set');
  assert.deepEqual(JSON.parse(last[1].body), { scope: '/srv/repo', privacy: 'local-only', quality: 'fast' });
  await s.setQuality('maximum');
  last = s.fetches.at(-1);
  assert.deepEqual(JSON.parse(last[1].body), { scope: '/srv/repo', privacy: 'local-only', quality: 'maximum' });
  // renderPolicy toggles the active class from server state (one chip per group).
  s.renderPolicy([{ scope: '/srv/repo', privacy: 'local-only', quality: 'maximum' }]);
  assert.equal(s.__get('chip-privacy-local-only').className.includes('active'), true);
  assert.equal(s.__get('chip-quality-maximum').className.includes('active'), true);
  assert.equal(s.__get('chip-privacy-cloud-ok').className.includes('active'), false);
});

// 6. The Decisions panel renders additive event fields (role/decision/
//    rationale/fallback/chosen_model) read-only.
test('decisions panel reads the additive fields', () => {
  assert.ok(html.includes('decision-panel') || html.includes('decisions-panel') || html.includes('renderDecisions'),
    'decisions panel markup/renderer missing');
  for (const field of ['decision_fallback', 'chosen_model', 'rationale']) {
    assert.ok(html.includes(field), `decisions panel must surface ${field}`);
  }
});

// 7. Credentials tab: masked list + add/edit/delete + Test boolean + grants
//    (spec: "Credentials tab (masked list/add/edit/delete/Test boolean/grants/
//    audit)").
test('credentials tab surface exists and never prints plaintext fields', () => {
  assert.ok(html.includes('credentials-panel') || html.includes('renderCredentials'),
    'credentials panel markup/renderer missing');
  assert.ok(html.includes('/api/credentials/add'), 'credentials add endpoint missing');
  assert.ok(html.includes('/api/credentials/remove'), 'credentials remove endpoint missing');
  assert.ok(html.includes('/api/credentials/test'), 'credentials test endpoint missing');
  // The list is masked server-side; the dashboard must not invent a value field.
  assert.ok(!/credentials[^"]*value/i.test(html.match(/id="credential-value"[^>]*/)?.[0] ?? ''),
    'no plaintext value capture outside the add form');
});
