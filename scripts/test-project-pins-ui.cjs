#!/usr/bin/env node
'use strict';

// Structural + behavioral checks for the Phase-2 per-project model pin UI in
// main_dashboard.html. Node's built-in test runner + a synthetic DOM via `vm`
// (no browser, no server, no deps), mirroring scripts/test-nav-ui.cjs.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../src/escalation/templates/main_dashboard.html'), 'utf8');

function ids() { return [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]); }
function requireOneId(id) { assert.equal(ids().filter(x => x === id).length, 1, `expected exactly one #${id}`); }
function tagForId(id) {
  const re = new RegExp(`<([a-zA-Z][\\w-]*)\\b[^>]*\\bid="${id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"[^>]*>`, 'm');
  const m = html.match(re);
  assert.ok(m, `missing #${id}`);
  return m[0];
}
function extractFn(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
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
    this.children = []; this._html = null;
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
    PIN_GLOBAL: '__global__',
    LOCAL_CUSTOM: '__custom__',
    pinScopeKey: '__global__',
    pinScopeResolved: true,
    pinEntry: null,
    pinGlobalProfile: null,
    pinResolveGen: 0,
    pinRoleInherit: { main: true, reviewer: true, subagent: true },
    routingProfile: {},
    routingProfileBusy: false,
    routingLocalModelIds: [],
    buildLocalModelSelect() {},
    renderRoutingChoice() {},
    document: { getElementById: id => elements.get(id) || null, createElement: tag => new El(tag) },
    ...extra,
  };
  vm.createContext(s);
  vm.runInContext(functionNames.map(extractFn).join('\n'), s, { filename: 'project-pins-ui.js' });
  s.__get = id => { if (!elements.has(id)) elements.set(id, new El('div', id)); return elements.get(id); };
  return s;
}

test('scope selector, per-role inherit controls, and pins list exist exactly once', () => {
  for (const id of [
    'pin-scope', 'pin-scope-add', 'pin-scope-add-btn', 'pin-scope-state',
    'main-inherit', 'reviewer-inherit', 'subagent-inherit',
    'main-inherit-wrap', 'reviewer-inherit-wrap', 'subagent-inherit-wrap',
    'project-pins-list', 'project-pins-list-wrap',
  ]) requireOneId(id);
  assert.ok(tagForId('pin-scope').includes('onchange="onPinScopeChange()"'));
});

test('the Save button keeps its Phase-1 handler and dispatches by scope', () => {
  // Preserving onclick="saveRoutingProfile()" keeps test-nav-ui.cjs green; the
  // function itself dispatches to the project path.
  assert.ok(tagForId('save-routing-profile').includes('onclick="saveRoutingProfile()"'));
  const save = extractFn('saveRoutingProfile');
  assert.ok(save.includes('isProjectScope()') && save.includes('saveProjectPin()'), 'save dispatches by scope');
});

test('no duplicate concrete ids after adding the pin UI', () => {
  const seen = new Map();
  for (const id of ids().filter(x => !x.includes('${'))) seen.set(id, (seen.get(id) || 0) + 1);
  assert.deepEqual([...seen].filter(([, n]) => n > 1), []);
});

test('describePin distinguishes pinned vs inherited roles', () => {
  const s = sandbox(['describePin']);
  assert.equal(
    s.describePin({ main: { backend: 'local', model: 'm' }, reviewer: null, subagent: 'pool' }),
    'main local/m · reviewer inherited · subagent pool',
  );
  assert.equal(
    s.describePin({ main: null, reviewer: { backend: 'cloud' }, subagent: null }),
    'main inherited · reviewer cloud · subagent inherited',
  );
});

test('single Save predicate: unresolved scope and empty pinned subagent disable Save', () => {
  const s = sandbox(['updateSaveState', 'isProjectScope', 'routingModelValue', 'routingLocalModelChanged'], {
    pinScopeKey: '/proj', pinScopeResolved: true, routingProfileBusy: false,
    pinRoleInherit: { main: true, reviewer: true, subagent: false },
  });
  const btn = s.__get('save-routing-profile');
  s.__get('subagent-model-select').value = '';        // pinned subagent, no id yet
  s.routingLocalModelChanged('subagent');             // real onchange path
  assert.equal(btn.disabled, true, 'pinned-but-empty subagent disables Save');
  s.__get('subagent-model-select').value = 'pool-a';  // pick a concrete id via the dropdown
  s.routingLocalModelChanged('subagent');             // must re-run the predicate (regression guard)
  assert.equal(btn.disabled, false, 'a valid pinned subagent enables Save');
  s.pinScopeResolved = false;                          // still resolving / not pinnable
  s.updateSaveState();
  assert.equal(btn.disabled, true, 'an unresolved scope disables Save');
});

test('editing one role leaves the others inherited (no clobber)', () => {
  const s = sandbox(['onRoleInherit', 'applyRoleInherit', 'updateSaveState', 'isProjectScope', 'routingModelValue', 'globalRoleChoice'], {
    pinScopeKey: '/proj', pinGlobalProfile: { main: { backend: 'local', model: 'g-main' }, reviewer: { backend: 'cloud' }, subagent_model: null },
    pinRoleInherit: { main: true, reviewer: true, subagent: true },
  });
  s.__get('reviewer-inherit').checked = false; // user pins only the reviewer
  s.onRoleInherit('reviewer');
  assert.deepEqual(s.pinRoleInherit, { main: true, reviewer: false, subagent: true }, 'only reviewer becomes pinned');
});

test('the pins list renders untrusted paths and model ids as inert text', () => {
  const s = sandbox(['renderProjectPinsList', 'describePin', 'pinBasename']);
  const hostileKey = '/repo/<img src=x onerror=alert(1)>';
  const hostileModel = '</script><script>alert(2)</script>';
  s.renderProjectPinsList({ [hostileKey]: { main: null, reviewer: null, subagent: hostileModel } });
  const list = s.__get('project-pins-list');
  const texts = [];
  let usedInnerHtml = false;
  (function walk(node) {
    if (node.textContent) texts.push(node.textContent);
    if (node.title) texts.push(node.title);
    if (node._html) usedInnerHtml = true;
    (node.children || []).forEach(walk);
  })(list);
  const joined = texts.join('\n');
  assert.ok(joined.includes('<img src=x onerror=alert(1)>'), 'hostile path present as literal text');
  assert.ok(joined.includes('</script><script>alert(2)</script>'), 'hostile model present as literal text');
  assert.ok(!usedInnerHtml, 'no innerHTML assignment anywhere in the list DOM');
  // Source-level guard: the renderer must use textContent, not innerHTML.
  const src = extractFn('renderProjectPinsList');
  assert.ok(src.includes('.textContent'), 'renderProjectPinsList uses textContent');
  assert.ok(!src.includes('.innerHTML'), 'renderProjectPinsList must not use innerHTML');
});
