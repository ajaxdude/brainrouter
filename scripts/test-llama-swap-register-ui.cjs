#!/usr/bin/env node
'use strict';

// UI-behavior test for registering completed llama_cpp download jobs into
// llama-swap. Extracts real dashboard helpers in a node:vm sandbox — no server,
// model, or network.

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(
  path.join(__dirname, '../src/escalation/templates/main_dashboard.html'),
  'utf8',
);

const escHtmlMatches = [...html.matchAll(/function escHtml\(s\)\s*\{[\s\S]*?\n\}/g)];
assert.equal(escHtmlMatches.length, 1, 'expected exactly one escHtml definition');
const escHtmlSrc = escHtmlMatches[0][0];

function extractFunction(name, async = false) {
  const start = html.indexOf(`${async ? 'async ' : ''}function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
  const brace = html.indexOf('{', start);
  let depth = 0;
  for (let i = brace; i < html.length; i++) {
    if (html[i] === '{') depth++;
    if (html[i] === '}') depth--;
    if (depth === 0) return html.slice(start, i + 1);
  }
  throw new Error(`unterminated function ${name}`);
}

function makeRenderSandbox() {
  const sandbox = { DOWNLOAD_STATUS_COLORS: { complete: 'green', downloading: 'cyan' }, TOOLBOX_BACKEND_LABELS: { llama_cpp: 'llama.cpp', ds4: 'DeepSeek' } };
  vm.createContext(sandbox);
  vm.runInContext(escHtmlSrc + '\n' + extractFunction('renderToolboxDownloadJobRow'), sandbox, { filename: 'register-render.js' });
  return sandbox;
}

test('Register button renders only for completed llama_cpp download-job rows with quant_pattern', () => {
  const s = makeRenderSandbox();

  assert.doesNotMatch(s.renderToolboxDownloadJobRow({ backend: 'llama_cpp', model_id: 'm', kind: 'download', status: 'downloading', quant_pattern: 'Q4.gguf', message: '' }), /Register in llama-swap/);
  assert.doesNotMatch(s.renderToolboxDownloadJobRow({ backend: 'llama_cpp', model_id: 'm', kind: 'download', status: 'complete', message: '' }), /Register in llama-swap/);
  assert.doesNotMatch(s.renderToolboxDownloadJobRow({ backend: 'ds4', model_id: 'm', kind: 'download', status: 'complete', quant_pattern: 'Q4.gguf', message: '' }), /Register in llama-swap/);
  assert.doesNotMatch(s.renderToolboxDownloadJobRow({ backend: 'llama_cpp', model_id: 'm', kind: 'prepare_ple', status: 'complete', quant_pattern: 'Q4.gguf', message: '' }), /Register in llama-swap/);

  const row = s.renderToolboxDownloadJobRow({ backend: 'llama_cpp', model_id: 'llama-model', kind: 'download', status: 'complete', quant_pattern: 'Q4.gguf', message: 'done' });
  assert.match(row, /Register in llama-swap/);
  assert.match(row, /registerInLlamaSwap\("llama-model", "Q4\.gguf"\)/);
});

test('registerInLlamaSwap posts typed body to the llama-swap registration endpoint', async () => {
  const calls = [];
  const alerts = [];
  const sandbox = {
    fetch: async (url, opts) => {
      calls.push({ url, opts });
      return { ok: true, json: async () => ({ status: 'registered', llama_swap_id: 'llama-model', message: 'Registered baseline' }) };
    },
    alert: (msg) => alerts.push(msg),
    loadToolboxModels: () => {},
  };
  vm.createContext(sandbox);
  vm.runInContext(extractFunction('registerInLlamaSwap', true), sandbox, { filename: 'register-action.js' });
  await sandbox.registerInLlamaSwap('llama-model', 'Q4.gguf');

  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/llama-swap/register-model');
  assert.equal(calls[0].opts.method, 'POST');
  assert.deepEqual(JSON.parse(calls[0].opts.body), { model_id: 'llama-model', quant_pattern: 'Q4.gguf' });
  assert.match(alerts[0], /llama-model: Registered baseline/);
});

test('static wiring: endpoint and completed-job gate exist', () => {
  assert.ok(html.includes('function registerInLlamaSwap('), 'registerInLlamaSwap missing');
  assert.ok(html.includes("fetch('/api/llama-swap/register-model'"), 'registration endpoint fetch missing');
  assert.ok(html.includes("job.backend === 'llama_cpp' && job.kind === 'download' && job.status === 'complete' && job.quant_pattern"), 'completed llama_cpp job gate missing');
});
