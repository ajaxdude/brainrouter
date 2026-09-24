#!/usr/bin/env node
'use strict';

// Mandatory UI-behavior test for the `hf` preflight banner + empty-model
// hints (design: docs/design/hf-preflight-download-diagnostics.md, R8).
// Extracts the REAL escHtml plus the marker-delimited pure-helper block from
// main_dashboard.html and exercises them in a node:vm sandbox — no server,
// model, or network. Also runs static source assertions so the test fails if
// the banner mounts, loader wiring, or ds4/halogen branches are missing.

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

// --- Extract the marker-delimited pure-helper block (asserting unique markers) ---
const START = '// === hf-preflight pure helpers (unit-tested by scripts/test-hf-preflight-ui.cjs) ===';
const END = '// === end hf-preflight pure helpers ===';
assert.equal(html.split(START).length - 1, 1, 'expected exactly one helper-block start sentinel');
assert.equal(html.split(END).length - 1, 1, 'expected exactly one helper-block end sentinel');
const blockSrc = html.slice(html.indexOf(START) + START.length, html.indexOf(END));

function makeSandbox() {
  const elements = new Map([
    ['toolbox-hf-banner', { innerHTML: '' }],
    ['server-mode-hf-banner', { innerHTML: '' }],
  ]);
  const sandbox = {
    toolboxHfPreflight: null,
    document: { getElementById: (id) => elements.get(id) || null },
  };
  vm.createContext(sandbox);
  vm.runInContext(escHtmlSrc + '\n' + blockSrc, sandbox, { filename: 'hf-helpers.js' });
  sandbox.__elements = elements;
  return sandbox;
}

test('hfFromStatus: success propagates hf, failed fetch preserves previous', () => {
  const s = makeSandbox();
  const hf = { found_on_path: false };
  assert.equal(s.hfFromStatus({ hf }, null), hf);
  const prev = { found_on_path: false, binary: 'hf' };
  assert.equal(s.hfFromStatus(null, prev), prev, 'null fetch must preserve last-known');
  assert.equal(s.hfFromStatus({}, prev), null, 'success without hf clears to null');
});

test('hfBannerHtml: shown only when missing, escapes untrusted content', () => {
  const s = makeSandbox();
  assert.equal(s.hfBannerHtml(null), '');
  assert.equal(s.hfBannerHtml({ found_on_path: true }), '');
  const cmd = 'python3 -m pip install --user -U "huggingface_hub[cli]"';
  const shown = s.hfBannerHtml({ found_on_path: false, message: 'nope', install_command: cmd });
  assert.match(shown, /⚠/);
  assert.ok(shown.includes('nope'));
  assert.ok(shown.includes('huggingface_hub[cli]'));
  // untrusted content must be HTML-escaped (element-content context)
  const evil = s.hfBannerHtml({ found_on_path: false, message: '<script>x', install_command: '<b>' });
  assert.ok(!evil.includes('<script>'), 'message must be escaped');
  assert.ok(!evil.includes('<b>'), 'install_command must be escaped');
  assert.ok(evil.includes('&lt;script&gt;'));
  assert.ok(evil.includes('&lt;b&gt;'));
});

test('serverModeEmptyModelHint / emptyModelOption switch on hf state', () => {
  const s = makeSandbox();
  for (const kind of ['ds4 models', 'halogen bundles']) {
    assert.match(s.serverModeEmptyModelHint(kind, { found_on_path: false }), /Hugging Face CLI/);
    assert.equal(
      s.serverModeEmptyModelHint(kind, { found_on_path: true }),
      'No downloaded ' + kind + ' — use Downloads to fetch one first',
    );
    const opt = s.emptyModelOption(kind, { found_on_path: true });
    assert.ok(opt.startsWith('<option value="">') && opt.endsWith('</option>'));
    assert.ok(opt.includes('No downloaded ' + kind));
  }
});

test('renderHfBanner sets banner when missing and clears it when found', () => {
  const s = makeSandbox();
  const el = s.__elements.get('toolbox-hf-banner');
  s.toolboxHfPreflight = { found_on_path: false, message: 'm', install_command: 'c' };
  s.renderHfBanner('toolbox-hf-banner');
  assert.ok(el.innerHTML.length > 0, 'banner should render when hf missing');
  s.toolboxHfPreflight = { found_on_path: true };
  s.renderHfBanner('toolbox-hf-banner');
  assert.equal(el.innerHTML, '', 'banner should clear when hf found');
  // unknown element id must be a no-op, not a throw
  s.renderHfBanner('does-not-exist');
});

test('static wiring: mounts, banner calls, loader assignment, ds4/halogen branches', () => {
  assert.ok(html.includes('<div id="toolbox-hf-banner"'), 'toolbox banner mount missing');
  assert.ok(html.includes('<div id="server-mode-hf-banner"'), 'server-mode banner mount missing');
  assert.ok(html.includes("renderHfBanner('toolbox-hf-banner')"), 'toolbox renderHfBanner call missing');
  assert.ok(html.includes("renderHfBanner('server-mode-hf-banner')"), 'server-mode renderHfBanner call missing');
  assert.equal(
    html.split('hfFromStatus(statusRes, toolboxHfPreflight)').length - 1,
    2,
    'both loaders must assign via hfFromStatus',
  );
  assert.ok(html.includes("emptyModelOption('ds4 models', toolboxHfPreflight)"), 'ds4 empty branch not wired');
  assert.ok(html.includes("emptyModelOption('halogen bundles', toolboxHfPreflight)"), 'halogen empty branch not wired');
});
