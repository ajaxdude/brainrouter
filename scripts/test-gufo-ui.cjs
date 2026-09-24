#!/usr/bin/env node
'use strict';

// Mandatory UI-behavior test for the gufo Server Mode / Downloads wiring
// (design: docs/design/gufo-backend-integration.md, DI-7 / T-5). Mirrors
// scripts/test-hf-preflight-ui.cjs: static source assertions that the gufo
// tabs, panel, mounts, and wiring exist, plus behavioral tests that extract
// the real gufo render functions from main_dashboard.html and exercise them
// in a node:vm sandbox with fixture data — no server, model, or network.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(
  path.join(__dirname, '../src/escalation/templates/main_dashboard.html'),
  'utf8',
);

// --- Extract the real escHtml (asserting it is unique) ---
const escHtmlMatches = [...html.matchAll(/function escHtml\(s\)\s*\{[\s\S]*?\n\}/g)];
assert.equal(escHtmlMatches.length, 1, 'expected exactly one escHtml definition');
const escHtmlSrc = escHtmlMatches[0][0];

function extractFn(name) {
  const re = new RegExp(`function ${name}\\(\\)\\s*\\{[\\s\\S]*?\\n\\}`, 'g');
  const matches = [...html.matchAll(re)];
  assert.equal(matches.length, 1, `expected exactly one ${name} definition`);
  return matches[0][0];
}

// Build a vm sandbox with a mock DOM element for `sm-gufo-model` plus the
// globals the gufo render functions read, then run escHtml + the extracted
// functions in it.
function makeSandbox({ modelsData, completeIds }) {
  const modelEl = { value: '', innerHTML: '' };
  const statusEl = { value: '', innerHTML: '' };
  const elements = new Map([
    ['sm-gufo-model', modelEl],
    ['sm-gufo-status', statusEl],
  ]);
  const sandbox = {
    toolboxModelsData: modelsData,
    toolboxHfPreflight: null,
    serverModeGufoStatus: null,
    presenceFor: (backend, id) =>
      completeIds.includes(id) ? { completeness: { kind: 'complete' } } : { completeness: { kind: 'incomplete' } },
    emptyModelOption: (label) => `<<EMPTY:${label}>>`,
    document: { getElementById: (id) => elements.get(id) || null },
  };
  vm.createContext(sandbox);
  vm.runInContext(
    [escHtmlSrc, extractFn('renderServerModeGufoModelOptions'), extractFn('renderServerModeGufoStatus')].join('\n'),
    sandbox,
    { filename: 'gufo-ui.js' },
  );
  sandbox.__modelEl = modelEl;
  sandbox.__statusEl = statusEl;
  return sandbox;
}

const MODELS = {
  backends: [
    {
      backend: 'gufo',
      entries: [
        { id: 'gufo-main', name: 'Qwen Main', payload: { role: 'main', speculative: { mode: 'dflash2', draft_model_id: 'gufo-draft' } } },
        { id: 'gufo-draft', name: 'DFlash2 draft', payload: { role: 'draft' } },
        { id: 'gufo-ar', name: 'AR-only Main', payload: { role: 'main' } },
      ],
    },
  ],
};

test('gufo model dropdown: main incomplete → not listed (empty hint)', () => {
  const s = makeSandbox({ modelsData: MODELS, completeIds: [] });
  s.renderServerModeGufoModelOptions();
  assert.equal(s.__modelEl.innerHTML, '<<EMPTY:gufo models>>');
});

test('gufo model dropdown: main complete but DFlash2 draft missing → draft-missing hint', () => {
  const s = makeSandbox({ modelsData: MODELS, completeIds: ['gufo-main'] });
  s.renderServerModeGufoModelOptions();
  assert.match(s.__modelEl.innerHTML, /DFlash2 draft/);
  assert.doesNotMatch(s.__modelEl.innerHTML, /<option value="gufo-main"/);
});

test('gufo model dropdown: main + draft complete → main is listed', () => {
  const s = makeSandbox({ modelsData: MODELS, completeIds: ['gufo-main', 'gufo-draft'] });
  s.renderServerModeGufoModelOptions();
  assert.match(s.__modelEl.innerHTML, /<option value="gufo-main">/);
  // The draft itself is never offered as a servable model.
  assert.doesNotMatch(s.__modelEl.innerHTML, /<option value="gufo-draft"/);
});

test('gufo model dropdown: an AR main (no speculative) is listed once complete', () => {
  const s = makeSandbox({ modelsData: MODELS, completeIds: ['gufo-ar'] });
  s.renderServerModeGufoModelOptions();
  assert.match(s.__modelEl.innerHTML, /<option value="gufo-ar">/);
});

test('gufo status renders running/stopped/absent', () => {
  const s = makeSandbox({ modelsData: MODELS, completeIds: [] });
  s.serverModeGufoStatus = { exists: false };
  s.renderServerModeGufoStatus();
  assert.match(s.__statusEl.innerHTML, /Not running/);

  s.serverModeGufoStatus = { exists: true, running: true, model_id: 'gufo-main', container_name: 'brainrouter-gufo-server' };
  s.renderServerModeGufoStatus();
  assert.match(s.__statusEl.innerHTML, /running/);
  assert.match(s.__statusEl.innerHTML, /gufo-main/);
});

test('static wiring: gufo is a first-class backend across arrays, panel, and loader', () => {
  // Backend arrays + label.
  assert.match(html, /const DOWNLOAD_CAPABLE_BACKENDS = \[[^\]]*'gufo'/, 'gufo missing from DOWNLOAD_CAPABLE_BACKENDS');
  assert.match(html, /const SERVER_MODE_CAPABLE_BACKENDS = \[[^\]]*'gufo'/, 'gufo missing from SERVER_MODE_CAPABLE_BACKENDS');
  assert.match(html, /gufo:\s*'Gufo'/, 'gufo missing from TOOLBOX_BACKEND_LABELS');

  // Server Mode panel + every sm-gufo-* control mount.
  for (const id of [
    'sm-panel-gufo',
    'sm-gufo-toolbox',
    'sm-gufo-model',
    'sm-gufo-ctx',
    'sm-gufo-sessions',
    'sm-gufo-host',
    'sm-gufo-port',
    'sm-gufo-custom-args',
    'sm-gufo-status',
  ]) {
    assert.ok(html.includes(`id="${id}"`), `dashboard is missing mount #${id}`);
  }

  // Status var + fetch + destructure/assign + panel show/hide.
  assert.ok(html.includes('let serverModeGufoStatus = null;'), 'gufo status state var missing');
  assert.ok(html.includes("safeFetch('/api/server-mode/gufo/status')"), 'gufo status fetch missing from loadServerMode');
  assert.ok(html.includes('if (gufoStatusRes) serverModeGufoStatus = gufoStatusRes;'), 'gufo status assignment missing');
  assert.match(html, /sm-panel-gufo'\)\.style\.display = backend === 'gufo'/, 'gufo panel show/hide missing from setServerModeBackend');

  // Render + action functions are defined and wired into loadServerMode.
  for (const fn of [
    'renderServerModeGufoToolboxOptions',
    'renderServerModeGufoModelOptions',
    'renderServerModeGufoStatus',
    'startGufoServer',
    'stopGufoServer',
  ]) {
    assert.ok(html.includes(`function ${fn}(`), `gufo function ${fn} not defined`);
  }
  assert.ok(html.includes('renderServerModeGufoModelOptions();'), 'gufo render not called by loadServerMode');
  assert.ok(html.includes("emptyModelOption('gufo models', toolboxHfPreflight)"), 'gufo empty-model branch not wired');

  // Start/stop hit the gufo endpoints.
  assert.ok(html.includes("fetch('/api/server-mode/gufo/start'"), 'gufo start endpoint call missing');
  assert.ok(html.includes("fetch('/api/server-mode/gufo/stop'"), 'gufo stop endpoint call missing');
});
