#!/usr/bin/env node
'use strict';

// Synthetic browser contracts only: no server, model, benchmark, or dependencies.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');
const html = fs.readFileSync(path.join(__dirname, '../src/escalation/templates/benchmarks.html'), 'utf8');
const script = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)][0][1];

class Element {
  constructor(tag, document) {
    this.tagName = tag.toUpperCase();
    this.document = document;
    this.children = [];
    this.listeners = {};
    this.attributes = {};
    this._text = '';
    this.value = '';
    this.disabled = false;
    this.hidden = false;
    this.open = false;
    this.isConnected = true;
    this.files = [];
    this.classList = { toggle: (name, enabled) => { this.attributes[name] = enabled; } };
  }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(x => x.textContent).join(''); }
  set innerHTML(_) { throw new Error('HTML insertion is prohibited'); }
  get options() { return this.children; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this._text = ''; this.children = nodes; }
  remove(index) { if (index !== undefined) this.children.splice(index, 1); }
  addEventListener(name, handler) { (this.listeners[name] ||= []).push(handler); }
  dispatch(name, extra = {}) {
    const event = { target: this, preventDefault() { this.prevented = true; }, ...extra };
    for (const handler of this.listeners[name] || []) handler(event);
    return this['on' + name]?.(event);
  }
  click() { if (!this.disabled) return this.dispatch('click'); }
  focus() { this.document.activeElement = this; }
  scrollIntoView() {}
  showModal() { this.open = true; this.document.getElementById('detail-close').focus(); }
  close() { this.open = false; this.dispatch('close'); }
}

const clone = value => JSON.parse(JSON.stringify(value));
const deferred = () => { let resolve, reject; const promise = new Promise((yes, no) => { resolve = yes; reject = no; }); return { promise, resolve, reject }; };
const response = (body, status = 200) => ({ ok: status < 400, status, json: async () => body, text: async () => typeof body === 'string' ? body : JSON.stringify(body) });
const emptyPage = { items: [], total: 0, total_pages: 0, page: 1 };
const filters = { families: [], backends: [], workloads: [], quant_names: [], statuses: [] };

function browser(handler = async () => response({}), settings = {}) {
  const nodes = new Map();
  const document = {
    getElementById(id) { assert(nodes.has(id), 'Unknown element: ' + id); return nodes.get(id); },
    createElement(tag) { return new Element(tag, document); },
    createElementNS(_, tag) { return new Element(tag, document); },
  };
  document.body = new Element('body', document);
  document.activeElement = document.body;
  for (const match of html.matchAll(/<([a-z][\w-]*)\b[^>]*\bid="([^"]+)"[^>]*>/g)) {
    const node = new Element(match[1], document), tag = match[0];
    node.value = /\bvalue="([^"]*)"/.exec(tag)?.[1] || '';
    node.disabled = /\bdisabled\b/.test(tag);
    node.hidden = /\bhidden\b/.test(tag);
    nodes.set(match[2], node);
  }
  for (const [id, value] of Object.entries({ sort: 'started_at:desc', 'memory-metric': 'peak_rss_bytes', repetition: '0', 'prepare-mode': 'metadata', 'plan-format': 'json' })) nodes.get(id).value = value;
  const calls = [], downloads = [], storage = new Map(), storageCalls = [], windowListeners = new Map();
  const sandbox = {
    document, console, URLSearchParams, TextEncoder, AbortController, Blob, setTimeout, clearTimeout,
    URL: { createObjectURL(blob) { downloads.push(blob); return 'blob:synthetic'; }, revokeObjectURL() {} },
    location: { href: '', hash: settings.hash || '', search: settings.search || '' },
    localStorage: {
      setItem(key, value) { storageCalls.push('save'); storage.set(key, value); },
      getItem(key) { storageCalls.push('load'); return storage.get(key) ?? null; },
      removeItem(key) { storageCalls.push('clear'); storage.delete(key); },
    },
    async fetch(address, options) {
      calls.push({ url: address, options });
      if (settings.fetch) return settings.fetch(address, options);
      if (address.startsWith('/api/benchmarks/runs?')) return response(emptyPage);
      if (address === '/api/benchmarks/filters') return settings.filtersResponse || response(filters);
      return handler(address, options);
    },
    addEventListener(type, handler) { windowListeners.set(type, handler); },
  };
  sandbox.window = sandbox;
  vm.createContext(sandbox);
  vm.runInContext(script, sandbox, { filename: 'benchmarks.html' });
  return { sandbox, nodes, document, calls, downloads, storageCalls, dispatchWindow: type => windowListeners.get(type)?.(), evaluate: code => vm.runInContext(code, sandbox) };
}

function fixture() {
  const attack = 'javascript:alert(1)"><img src=x onerror=alert(2)> /private/file.sh; $(run)';
  return {
    run: { run_id: 'run-synthetic', status: 'succeeded' },
    run_record: { id: 'run-synthetic', exact_command: attack, stdout_path: attack, stderr_path: attack, cwd: attack, environment: { PATH: attack }, status: 'succeeded' },
    configuration: {
      model: { id: 'model', family: 'Synthetic', parameter_count_total: 1000, model_kind: 'dense' },
      artifact: { id: 'artifact', disk_bytes: 1073741824, source_uri: attack, conversion_command: attack, sha256: 'a'.repeat(64) },
      runtime: { id: 'runtime', repository: attack, capabilities: { flash_attention: 'requested_unavailable' } },
      hardware: { id: 'hardware', unified_memory: true, system_ram_bytes: 17179869184, gpus: [{ index: 2, name: 'GPU two' }, { index: 0, name: 'GPU zero' }] },
      workload: { id: 'workload', name: 'Synthetic tasks' },
      experiment: { id: 'experiment', command_template: attack, context_tokens: 8192, prompt_tokens: 512, generation_tokens: 128, optimization: { ngram: 'enabled', ngram_storage_location: 'cpu', speculator_type: 'ngram' }, sampling: { seed: 0, temperature: 0, top_p: 1 } },
    },
    performance_metrics: { model_load_ms: 0, prompt_processing_ms: 2, generation_ms: 3, ttft_ms: null, prompt_tps: 4, generation_tps: 5, inter_token_p50_ms: 1, inter_token_p95_ms: 2, inter_token_p99_ms: 3, peak_rss_bytes: 1073741824, peak_vram_bytes: null, kv_cache_bytes: 1, energy_joules: 0, avg_power_watts: null },
    speculative_metrics: { acceptance_rate: 0, proposed_tokens: 0, accepted_tokens: 0, speculator_memory_bytes: null, overhead_ms: 0 },
    quality_results: [{ id: 'quality', task_id: 'task-a', metric_name: 'pass@1', metric_value: 0, passed: false, compile_succeeded: null, tests_passed: 0, tests_total: null, output_path: attack, log_path: attack }],
    telemetry_samples: [],
  };
}
function templateFixture() {
  return { model: { id: 'm' }, artifact: { id: 'a' }, runtime: { id: 'r' }, hardware: { id: 'h' }, workload: { id: 'w' }, experiment: { context_tokens: 8192, prompt_tokens: 512, generation_tokens: 128 } };
}
function walk(node) { return [node, ...node.children.flatMap(walk)]; }
function populatePreparation(app) {
  app.nodes.get('template-json').value = JSON.stringify(templateFixture());
  app.nodes.get('prepare-status').value = 'planned';
  app.nodes.get('exact-command').value = 'synthetic-command --not-executed';
}

test('telemetry preserves null gaps and zero, orders per GPU, and never sums duplicated host fields', () => {
  const app = browser();
  app.sandbox.samples = [
    { sampled_at: '2026-01-01T00:00:02Z', gpu_index: 2, gpu_util_percent: 90, cpu_percent: 12, rss_bytes: 100 },
    { sampled_at: '2026-01-01T00:00:00Z', gpu_index: 0, gpu_util_percent: 0, cpu_percent: 0, rss_bytes: 0, power_watts: 0, temperature_c: -2 },
    { sampled_at: '2026-01-01T00:00:00Z', gpu_index: 2, gpu_util_percent: 20, cpu_percent: 0, rss_bytes: 0 },
    { sampled_at: '2026-01-01T00:00:01Z', gpu_index: 0, gpu_util_percent: null, cpu_percent: 10, rss_bytes: 100 },
    { sampled_at: '2026-01-01T00:00:01Z', gpu_index: 2, gpu_util_percent: 30, cpu_percent: 10, rss_bytes: 100 },
    { sampled_at: '2026-01-01T00:00:03Z', gpu_index: 0, gpu_util_percent: 50, cpu_percent: 12 },
    { sampled_at: '2026-01-01T00:00:03Z', gpu_index: 2, gpu_util_percent: 60, cpu_percent: 16 },
    { sampled_at: 'bad-time', gpu_index: 4, gpu_util_percent: 0 },
    { sampled_at: '2026-01-01T00:00:02Z', gpu_index: null, power_watts: 0 },
  ];
  const result = app.evaluate(`(() => { const model=telemetryModel(samples); return {
    indices:model.groups.map(x=>x.index), cpu:telemetrySeries(model,model.host,'cpu_percent').map(x=>x.value),
    ram:telemetrySeries(model,model.host,'rss_bytes').map(x=>x.value),
    gpu0:telemetrySeries(model,model.groups[0].points,'gpu_util_percent'),
    conflicts:model.hostConflicts, invalid:model.invalidTimes
  }; })()`);
  assert.deepEqual(clone(result.indices), [0, 2, 'unattributed']);
  assert.deepEqual(clone(result.cpu), [0, 10, 12, null]);
  assert.deepEqual(clone(result.ram), [0, 100, 100, null]);
  assert.deepEqual(clone(result.gpu0.map(x => x.value)), [0, null, null, 50]);
  assert.equal(result.conflicts, 1);
  assert.equal(result.invalid, 1);
  app.sandbox.series = result.gpu0;
  const path = app.evaluate('sparkPath(series, t=>t/1000, v=>v)');
  assert.equal((path.match(/ M/g) || []).length, 2);
  assert.equal((path.match(/ L/g) || []).length, 0);
  assert.match(path, /0\.00/);
  assert.equal(app.evaluate('finite(null)'), false);
  assert.equal(app.evaluate('finite(0)'), true);
});

test('memory charts use one reported measure and quality excludes missing scores, not zero', () => {
  const app = browser();
  app.sandbox.rows = [
    { generation_tps: 0, peak_rss_bytes: 1073741824, peak_vram_bytes: null, quality_score: 0, disk_bytes: 1073741824 },
    { generation_tps: 10, peak_rss_bytes: null, peak_vram_bytes: 2147483648, quality_score: null, disk_bytes: 1073741824 },
    { generation_tps: null, peak_rss_bytes: 3, peak_vram_bytes: 4, quality_score: .5, disk_bytes: 0 },
  ];
  assert.deepEqual(clone(app.evaluate("memoryPoints(rows,'peak_rss_bytes').map(x=>x.memory_gib)")), [1]);
  assert.deepEqual(clone(app.evaluate("memoryPoints(rows,'peak_vram_bytes').map(x=>x.memory_gib)")), [2]);
  assert.deepEqual(clone(app.evaluate('qualityPoints(rows).map(x=>x.quality_per_gb)')), [0]);
  assert.equal(app.evaluate('bool(null)'), 'Not reported');
  assert.equal(app.evaluate('bool(false)'), 'No');
  assert.equal(app.evaluate('num(null)'), 'Not reported');
});

test('rich inspector renders malicious paths/commands as inert text and covers metrics, configuration, and quality', () => {
  const app = browser();
  app.sandbox.detailFixture = fixture();
  app.evaluate('renderDetail(detailFixture)');
  const root = app.nodes.get('detail-content'), text = root.textContent, all = walk(root);
  assert(text.includes(fixture().run_record.exact_command));
  for (const textLabel of ['Inter-token latency p50', 'Inter-token latency p95', 'Inter-token latency p99', 'Model load', 'Energy', 'Average power', 'Optimization', 'Sampling', 'Per-task quality evidence', '0.0000', 'Not reported', 'GPU 0', 'GPU 2']) assert(text.includes(textLabel), textLabel);
  assert.equal(all.filter(x => ['IMG', 'A', 'SCRIPT', 'IFRAME'].includes(x.tagName)).length, 0);
  assert(all.every(x => !Object.keys(x.attributes).some(name => /^(href|src|onerror)$/.test(name))));
  assert(!script.includes('innerHTML'));
  assert(!script.includes('insertAdjacentHTML'));
});

test('dialog opens immediately, rejects stale requests, reports 413 inline, closes on Escape and returns focus', async () => {
  const requests = [deferred(), deferred(), deferred()];
  let index = 0;
  const app = browser(() => requests[index++].promise);
  const trigger = app.document.createElement('button');
  trigger.focus();
  const first = app.sandbox.showDetail('first', trigger);
  assert.equal(app.nodes.get('detail').open, true);
  assert.match(app.nodes.get('detail-status').textContent, /Loading/);
  assert.equal(app.nodes.get('detail-content').attributes['aria-busy'], 'true');
  assert.equal(app.document.activeElement, app.nodes.get('detail-close'));
  const second = app.sandbox.showDetail('second');
  requests[1].resolve(response(fixture()));
  await second;
  requests[0].resolve(response({ stale: true }));
  await first;
  assert.equal(app.nodes.get('detail-title').textContent, 'Run second');
  assert(!app.nodes.get('detail-json').textContent.includes('stale'));
  assert.equal(app.nodes.get('detail-content').attributes['aria-busy'], 'false');
  app.nodes.get('detail').dispatch('cancel');
  assert.equal(app.nodes.get('detail').open, false);
  assert.equal(app.document.activeElement, trigger);
  const failure = app.sandbox.showDetail('oversize', trigger);
  requests[2].resolve(response({ error: 'Detail exceeds limit; no truncation.' }, 413));
  await failure;
  assert.match(app.nodes.get('detail-status').textContent, /HTTP 413/);
  assert.equal(app.nodes.get('detail-retry').hidden, false);
  assert.equal(app.nodes.get('detail-download').disabled, true);
});

test('closing a loading dialog cannot reopen it when its request completes', async () => {
  const pending = deferred(), app = browser(() => pending.promise);
  const request = app.sandbox.showDetail('late');
  app.nodes.get('detail').dispatch('cancel');
  pending.resolve(response(fixture()));
  await request;
  assert.equal(app.nodes.get('detail').open, false);
  assert.equal(app.nodes.get('detail-download').disabled, true);
});

test('preparation is nonmutating; confirmation posts exactly the preview, once, and offers a real Open run link', async () => {
  const bundle = { run: { id: 'deterministic-run', status: 'planned', repetition: 0 }, marker: 'server-normalized' };
  const app = browser(async url => url.endsWith('/ingest') ? response({ run_id: 'deterministic-run' }, 201) : response({ valid: true, persisted: false, bundle, warnings: ['Declared hash not verified'] }));
  populatePreparation(app);
  await app.sandbox.prepare();
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 0);
  assert.equal(app.nodes.get('confirm-import').disabled, false);
  assert.match(app.nodes.get('preview-warnings').textContent, /not verified/);
  const first = app.sandbox.confirmImport(), duplicate = app.sandbox.confirmImport();
  await Promise.all([first, duplicate]);
  const writes = app.calls.filter(x => x.url.endsWith('/ingest'));
  assert.equal(writes.length, 1);
  assert.deepEqual(JSON.parse(writes[0].options.body), bundle);
  const link = app.nodes.get('success').children.find(x => x.tagName === 'A');
  assert(link);
  assert.equal(link.textContent, 'Open run');
  assert.equal(link.href, '/benchmarks?run_id=deterministic-run');
  assert.equal(app.nodes.get('repetition').value, '0');
  assert.equal(app.nodes.get('confirm-import').disabled, true);
});

test('source changes invalidate preview, download, and confirmation; async preview races cannot restore them', async () => {
  const pending = deferred();
  const app = browser(() => pending.promise);
  populatePreparation(app);
  const preparation = app.sandbox.prepare();
  app.nodes.get('exact-command').value = 'edited-command';
  app.nodes.get('exact-command').dispatch('input');
  pending.resolve(response({ valid: true, persisted: false, bundle: { run: { id: 'stale' } }, warnings: [] }));
  await preparation;
  await app.sandbox.confirmImport();
  app.nodes.get('download-preview').click();
  assert.equal(app.nodes.get('confirm-import').disabled, true);
  assert.equal(app.nodes.get('download-preview').disabled, true);
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 0);
  assert.equal(app.downloads.length, 0);
  assert(!app.nodes.get('preview-json').textContent.includes('stale'));
});

test('preview fingerprint protects against programmatic edits even without input events', async () => {
  const app = browser(async () => response({ valid: true, persisted: false, bundle: { run: { id: 'preview', status: 'planned' } }, warnings: [] }));
  populatePreparation(app);
  await app.sandbox.prepare();
  app.nodes.get('repetition').value = '1';
  await app.sandbox.confirmImport();
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 0);
  assert.equal(app.nodes.get('confirm-import').disabled, true);
});

test('every source editor/control invalidates a ready preview and its raw download', async () => {
  const app = browser(async () => response({ valid: true, persisted: false, bundle: { run: { id: 'ready', status: 'planned' } }, warnings: [] }));
  for (const id of app.evaluate('sourceIds')) {
    populatePreparation(app);
    await app.sandbox.prepare();
    assert.equal(app.nodes.get('confirm-import').disabled, false, id + ' preview ready');
    app.nodes.get(id).dispatch('input');
    assert.equal(app.nodes.get('confirm-import').disabled, true, id + ' confirm invalidated');
    assert.equal(app.nodes.get('download-preview').disabled, true, id + ' download invalidated');
    assert.equal(app.nodes.get('preview-json').textContent, 'No current preview.');
    assert.equal(app.nodes.get('preview-response-json').textContent, 'No current validation response.');
  }
});

test('raw detail and validated bundle downloads preserve server JSON, never the unvalidated editor', async () => {
  const detail = fixture(), bundle = { run: { id: 'normalized', status: 'planned', repetition: 0 }, server: 'validated' };
  const app = browser(async url => response(url.endsWith('/prepare') ? { valid: true, persisted: false, bundle, warnings: [] } : detail));
  populatePreparation(app);
  await app.sandbox.prepare();
  app.nodes.get('download-preview').click();
  await app.sandbox.showDetail('raw-detail');
  app.nodes.get('detail-download').click();
  assert.equal(app.downloads.length, 2);
  assert.deepEqual(JSON.parse(await app.downloads[0].text()), bundle);
  assert.deepEqual(JSON.parse(await app.downloads[1].text()), detail);
});

test('raw inspector view and download preserve exact i64/u64 response text, whitespace, and numeric literals', async () => {
  const detail = fixture();
  detail.run_record.random_seed = '__SIGNED__';
  detail.configuration.artifact.disk_bytes = '__UNSIGNED__';
  detail.configuration.model.metadata = { large_string: '18446744073709551615' };
  const original = '\n' + JSON.stringify(detail, null, 3).replace('"__SIGNED__"', '-9223372036854775808').replace('"__UNSIGNED__"', '18446744073709551615') + '\n\n';
  const app = browser(async () => response(original));
  await app.sandbox.showDetail('exact-integers');
  assert.equal(app.nodes.get('detail-json').textContent, original);
  assert.match(app.nodes.get('detail-status').textContent, /outside the browser-safe range/);
  assert.match(app.nodes.get('detail-status').textContent, /Only the original raw JSON/);
  assert(app.nodes.get('detail-content').children.length > 0);
  assert.equal(app.nodes.get('detail-content').attributes['aria-busy'], 'false');
  app.nodes.get('detail-download').click();
  assert.equal(app.downloads.length, 1);
  assert.equal(await app.downloads[0].text(), original);
});

test('JSON safety recognizes signed integers, exponent notation, and boundary fractions without treating strings as numbers', () => {
  const app = browser();
  for (const token of ['9007199254740992', '9007199254740993', '-9223372036854775808', '18446744073709551615', '9.007199254740992e15', '-9.223372036854775808e18', '9007199254740991.0000000000001', '1e309']) {
    assert.throws(() => app.sandbox.parseInput('{"seed":' + token + '}', 'Synthetic input'), /safe integer range.*direct API.*metadata values as strings/, token);
  }
  for (const token of ['9007199254740991', '-9007199254740991', '9007199254740991.0', '90071992547409910e-1', '9.007199254740991e15', '0.001e18', '0', '0e999999', '-0.5']) {
    assert.doesNotThrow(() => app.sandbox.parseInput('{"value":' + token + '}', 'Synthetic input'), token);
  }
  const strings = { id: '18446744073709551615', metadata: { note: '"seed":9223372036854775807', value: '\\123456789012345678901234567890', exponential: '1e309' } };
  assert.deepEqual(clone(app.sandbox.parseInput(JSON.stringify(strings), 'String metadata')), strings);
});

test('unsafe template and bundle integer values are rejected before saving, preparing, validating, or rewriting input', async () => {
  const app = browser();
  populatePreparation(app);
  const template = JSON.stringify(templateFixture()).replace('"context_tokens":8192', '"context_tokens":9223372036854775807');
  app.nodes.get('template-json').value = template;
  app.sandbox.saveTemplate();
  assert.deepEqual(app.storageCalls, []);
  assert.match(app.nodes.get('template-status').textContent, /safe integer range/);
  await app.sandbox.prepare();
  assert.match(app.nodes.get('workflow-status').textContent, /direct API/);
  assert.equal(app.nodes.get('template-json').value, template);
  const bundle = '{"run":{"random_seed":-9223372036854775808}}';
  app.nodes.get('bundle-json').value = bundle;
  await app.sandbox.validateBundle();
  assert.match(app.nodes.get('workflow-status').textContent, /safe integer range/);
  assert.equal(app.nodes.get('bundle-json').value, bundle);
  assert.equal(app.calls.filter(x => x.options?.method === 'POST').length, 0);
});

test('unsafe llama-bench file numbers never become parsed result state', async () => {
  const app = browser();
  app.nodes.get('llama-file').files = [{ name: 'unsafe.json', size: 50, text: async () => '{"generation_tps":1,"n_prompt":18446744073709551615}' }];
  await app.sandbox.loadLlama();
  assert.match(app.nodes.get('llama-status').textContent, /safe integer range/);
  assert.equal(app.evaluate('llamaResult'), null);
  assert.equal(app.nodes.get('confirm-import').disabled, true);
});

test('unsafe server preview integers cannot enable confirmation, but the original response remains exact and read-only', async () => {
  const original = ' {\n "valid":true,"persisted":false,"bundle":{"run":{"id":"unsafe-preview","status":"planned","repetition":18446744073709551615}},"warnings":[]\n}\n';
  const app = browser(async () => response(original));
  populatePreparation(app);
  await app.sandbox.prepare();
  assert.equal(app.nodes.get('preview-response-json').textContent, original);
  assert.equal(app.nodes.get('confirm-import').disabled, true);
  assert.equal(app.nodes.get('download-preview').disabled, true);
  assert.equal(app.evaluate('prepared'), null);
  assert.match(app.nodes.get('workflow-status').textContent, /Validation response.*safe integer range/);
  await app.sandbox.confirmImport();
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 0);
});

test('request serialization refuses unsafe programmatic numbers before any persistence call', async () => {
  const app = browser(async () => response({ valid: true, persisted: false, bundle: { run: { id: 'safe-preview', status: 'planned', repetition: 0 } }, warnings: [] }));
  populatePreparation(app);
  await app.sandbox.prepare();
  app.evaluate('prepared.bundle.run.repetition=9007199254740992');
  await app.sandbox.confirmImport();
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 0);
  assert.match(app.nodes.get('success').textContent, /safe integer range/);
});

test('JSON matrix inputs and expanded JSON plans reject unsafe counts before candidates can be selected', async () => {
  const unsafePlan = '{"name":"unsafe","experiments":[],"exclusions":[],"repetitions":1,"run_count":9223372036854775807}';
  const app = browser(async () => response(unsafePlan));
  app.nodes.get('plan-source').value = '{"name":"unsafe","contexts":[18446744073709551615]}';
  await app.sandbox.previewPlan();
  assert.match(app.nodes.get('plan-status').textContent, /Experiment matrix.*safe integer range/);
  assert.equal(app.calls.filter(x => x.url.endsWith('/plan')).length, 0);
  app.nodes.get('plan-format').value = 'yaml';
  app.nodes.get('plan-source').value = 'name: unsafe\ncontexts: [18446744073709551615]\n';
  await app.sandbox.previewPlan();
  const call = app.calls.find(x => x.url.endsWith('/plan'));
  assert.equal(call.options.body, app.nodes.get('plan-source').value);
  assert.match(app.nodes.get('plan-status').textContent, /Server response.*safe integer range/);
  assert.equal(app.nodes.get('download-plan').disabled, true);
  assert.equal(app.nodes.get('plan-results').children.length, 0);
  assert.equal(app.evaluate('planData'), null);
});

test('an invalid preview response never enables confirmation', async () => {
  const app = browser(async () => response({ valid: true, persisted: true, bundle: { run: { id: 'wrong-contract' } } }));
  populatePreparation(app);
  await app.sandbox.prepare();
  assert.equal(app.nodes.get('confirm-import').disabled, true);
  assert.match(app.nodes.get('workflow-status').textContent, /non-persisted, validated bundle/);
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 0);
});

test('llama preparation requires an actual file and explicit completed status; metadata-only permits result-free planned runs', async () => {
  const app = browser(async () => response({ valid: true, persisted: false, bundle: { run: { id: 'result' } }, warnings: [] }));
  populatePreparation(app);
  app.nodes.get('prepare-mode').value = 'llama';
  await app.sandbox.prepare();
  assert.match(app.nodes.get('workflow-status').textContent, /actual llama-bench/);
  assert.equal(app.calls.filter(x => x.url.endsWith('/prepare')).length, 0);
  app.nodes.get('llama-file').files = [{ name: 'synthetic.json', size: 20, text: async () => '{"generation_tps":0}' }];
  await app.sandbox.loadLlama();
  await app.sandbox.prepare();
  assert.match(app.nodes.get('workflow-status').textContent, /completed status/);
  app.nodes.get('prepare-status').value = 'succeeded';
  await app.sandbox.prepare();
  const body = JSON.parse(app.calls.filter(x => x.url.endsWith('/prepare')).at(-1).options.body);
  assert.equal(body.llama_bench.generation_tps, 0);
  assert.equal(body.status, 'succeeded');
  app.nodes.get('prepare-mode').value = 'metadata';
  app.nodes.get('prepare-status').value = 'planned';
  app.nodes.get('prepare-status').dispatch('input');
  await app.sandbox.prepare();
  assert(!('llama_bench' in JSON.parse(app.calls.filter(x => x.url.endsWith('/prepare')).at(-1).options.body)));
});

test('bundle upload only fills editor; explicit validation uses correct endpoints and never immediately imports', async () => {
  const app = browser(async () => response({ valid: true, persisted: false, bundle: { run: { id: 'bundle-run' } }, warnings: [] }));
  app.nodes.get('bundle-file').files = [{ name: 'bundle.json', size: 50, text: async () => '{"run":{"status":"succeeded"}}' }];
  await app.nodes.get('bundle-file').dispatch('change');
  assert.equal(app.calls.filter(x => x.options?.method === 'POST').length, 0);
  await app.sandbox.validateBundle();
  assert(app.calls.some(x => x.url === '/api/benchmarks/validate'));
  app.nodes.get('llama-file').files = [{ name: 'result.json', size: 25, text: async () => '{"generation_tps":1}' }];
  await app.sandbox.loadLlama();
  await app.sandbox.validateBundle(true);
  assert(app.calls.some(x => x.url === '/api/benchmarks/validate/llama-bench'));
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 0);
});

test('metadata storage is explicit only and stale file reads cannot override manual edits', async () => {
  const app = browser();
  assert.deepEqual(app.storageCalls, []);
  populatePreparation(app);
  app.sandbox.saveTemplate();
  assert.deepEqual(app.storageCalls, ['save']);
  app.nodes.get('template-json').value = '';
  app.sandbox.loadSavedTemplate();
  assert.deepEqual(app.storageCalls, ['save', 'load']);
  assert.match(app.nodes.get('template-json').value, /model/);
  const pending = deferred();
  app.nodes.get('template-file').files = [{ name: 'slow.json', size: 2, text: () => pending.promise }];
  const loading = app.nodes.get('template-file').dispatch('change');
  app.nodes.get('template-json').value = '{"manual":true}';
  app.nodes.get('template-json').dispatch('input');
  pending.resolve('{"stale":true}');
  await loading;
  assert.equal(app.nodes.get('template-json').value, '{"manual":true}');
  app.sandbox.clearTemplate();
  assert.deepEqual(app.storageCalls, ['save', 'load', 'clear']);
  assert.equal(app.nodes.get('template-json').value, '');
});

test('file completion invalidates a preview prepared from the old editor while reading', async () => {
  const pending = deferred();
  const app = browser(async () => response({ valid: true, persisted: false, bundle: { run: { id: 'old-editor' } }, warnings: [] }));
  populatePreparation(app);
  app.nodes.get('bundle-file').files = [{ name: 'new.json', size: 2, text: () => pending.promise }];
  const loading = app.nodes.get('bundle-file').dispatch('change');
  await app.sandbox.prepare();
  assert.equal(app.nodes.get('confirm-import').disabled, false);
  pending.resolve('{"new":true}');
  await loading;
  assert.equal(app.nodes.get('confirm-import').disabled, true);
  assert.equal(app.nodes.get('download-preview').disabled, true);
});

test('plans render bounded pages, allow any eligible index, and pass the selected candidate to preparation', async () => {
  const candidate = { id: 'experiment', artifact_id: 'a', runtime_id: 'r', hardware_id: 'h', workload_id: 'w', context_tokens: 8192, prompt_tokens: 512, generation_tokens: 128, optimization: { speculator_type: 'none' } };
  const plan = { name: 'synthetic', experiments: Array.from({ length: 10000 }, (_, i) => ({ ...candidate, id: 'experiment-' + i })), exclusions: [{ reason_code: 'invalid', reason: 'Synthetic exclusion', candidate: { context_tokens: 1 } }], repetitions: 2, run_count: 20000 };
  const app = browser(async url => response(url.endsWith('/plan') ? plan : { valid: true, persisted: false, bundle: { run: { id: 'selected' } }, warnings: [] }));
  populatePreparation(app);
  app.nodes.get('plan-format').value = 'yaml';
  app.nodes.get('plan-source').value = 'name: synthetic';
  await app.sandbox.previewPlan();
  const planCall = app.calls.find(x => x.url.endsWith('/plan'));
  assert.equal(planCall.options.headers['content-type'], 'application/yaml');
  assert.equal(walk(app.nodes.get('plan-results')).filter(x => x.tagName === 'TR').length, 26);
  assert.match(app.nodes.get('plan-status').textContent, /10000 eligible/);
  app.sandbox.selectCandidate(9999);
  assert.match(app.nodes.get('selected-candidate').textContent, /experiment-9999/);
  assert.equal(app.nodes.get('repetition').value, '0');
  await app.sandbox.prepare();
  const body = JSON.parse(app.calls.filter(x => x.url.endsWith('/prepare')).at(-1).options.body);
  assert.equal(body.experiment.id, 'experiment-9999');
  app.nodes.get('plan-source').value = 'changed';
  app.nodes.get('plan-source').dispatch('input');
  assert.equal(app.nodes.get('download-plan').disabled, true);
  assert.equal(app.nodes.get('confirm-import').disabled, true);
});

test('failed or conflicting imports are visible and do not invent repetitions or silently retry', async () => {
  const app = browser(async url => url.endsWith('/ingest') ? response({ error: 'Successful run is immutable' }, 409) : response({ valid: true, persisted: false, bundle: { run: { id: 'conflict', status: 'succeeded', repetition: 7 } }, warnings: [] }));
  populatePreparation(app);
  app.nodes.get('repetition').value = '7';
  await app.sandbox.prepare();
  await app.sandbox.confirmImport();
  assert.match(app.nodes.get('success').textContent, /HTTP 409/);
  assert.equal(app.nodes.get('repetition').value, '7');
  assert.equal(app.nodes.get('confirm-import').disabled, true);
  assert.equal(app.calls.filter(x => x.url.endsWith('/ingest')).length, 1);
});

test('legacy run-link fragments work when opened separately and encode IDs as API path segments', async () => {
  const app = browser(async () => response(fixture()));
  app.sandbox.location.hash = '#run=' + encodeURIComponent('run / unsafe?');
  app.sandbox.openLinkedRun();
  assert(app.calls.some(x => x.url === '/api/benchmarks/runs/run%20%2F%20unsafe%3F'));
  assert.equal(app.nodes.get('detail').open, true);
});

test('initial run_id query opens the loading dialog independently of failed filter loading', async () => {
  const pending = deferred(), id = 'run / unsafe? &+=%2F<script>';
  const app = browser(() => pending.promise, {
    search: '?' + new URLSearchParams({ run_id: id }),
    filtersResponse: response({ error: 'Filters unavailable' }, 503),
  });
  assert.equal(app.nodes.get('detail').open, true);
  assert.match(app.nodes.get('detail-status').textContent, /Loading run detail/);
  assert.equal(app.nodes.get('detail-title').textContent, 'Run ' + id);
  assert(app.calls.some(x => x.url === '/api/benchmarks/runs/' + encodeURIComponent(id)));
  await new Promise(resolve => setImmediate(resolve));
  assert(!app.calls.some(x => x.url === '/api/benchmarks/filters'));
  assert.equal(app.nodes.get('detail').open, true);
  assert.match(app.nodes.get('detail-status').textContent, /Loading run detail/);
  pending.resolve(response(fixture()));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(app.nodes.get('detail-download').disabled, false);
  assert.match(app.nodes.get('message').textContent, /Filters unavailable/);
});

test('deep-link startup prioritizes detail and never exceeds two concurrent requests', async () => {
  const pending = new Map();
  let active = 0, peak = 0, rejected = 0;
  const app = browser(undefined, {
    search: '?run_id=priority-run',
    fetch(address) {
      active++;
      peak = Math.max(peak, active);
      if (active > 2) {
        rejected++;
        active--;
        return response({ error: 'Only two active requests are permitted' }, 503);
      }
      const request = deferred();
      pending.set(address, request);
      return request.promise.finally(() => { active--; });
    },
  });
  const detailURL = '/api/benchmarks/runs/priority-run', filterURL = '/api/benchmarks/filters';
  const runsURL = app.calls.find(x => x.url.startsWith('/api/benchmarks/runs?')).url;
  assert.equal(app.calls[0].url, detailURL);
  assert.equal(app.calls[1].url, runsURL);
  assert.equal(active, 2);
  assert.equal(app.nodes.get('detail').open, true);
  assert.match(app.nodes.get('detail-status').textContent, /Loading/);
  assert(!pending.has(filterURL));
  pending.get(detailURL).resolve(response(fixture()));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(app.nodes.get('detail-download').disabled, false);
  assert(pending.has(filterURL));
  assert.equal(active, 2); // The runs request is still pending while filters load.
  pending.get(filterURL).resolve(response(filters));
  pending.get(runsURL).resolve(response(emptyPage));
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(active, 0);
  assert.equal(peak, 2);
  assert.equal(rejected, 0);
  assert.equal(app.calls.length, 3);
  assert(!app.calls.some(x => x.options?.method === 'POST'));
});

test('a failed deep-link detail releases startup filter loading without retrying the request', async () => {
  const pending = deferred();
  const app = browser(() => pending.promise, { search: '?run_id=missing-run' });
  assert.equal(app.nodes.get('detail').open, true);
  assert(!app.calls.some(x => x.url === '/api/benchmarks/filters'));
  pending.resolve(response({ error: 'Run not found' }, 404));
  await new Promise(resolve => setImmediate(resolve));
  assert.match(app.nodes.get('detail-status').textContent, /HTTP 404/);
  assert.equal(app.calls.filter(x => x.url === '/api/benchmarks/runs/missing-run').length, 1);
  assert.equal(app.calls.filter(x => x.url === '/api/benchmarks/filters').length, 1);
});

test('initial query IDs take precedence over legacy fragments and empty IDs do not open a dialog', () => {
  const app = browser(async () => response(fixture()), { search: '?run_id=query-id', hash: '#run=fragment-id' });
  assert(app.calls.some(x => x.url === '/api/benchmarks/runs/query-id'));
  assert(!app.calls.some(x => x.url === '/api/benchmarks/runs/fragment-id'));
  const empty = browser(undefined, { search: '?run_id=' });
  assert.equal(empty.nodes.get('detail').open, false);
});

test('workflow and skip anchor navigation never reopens an initial query run, while run fragments still work', async () => {
  const app = browser(async () => response(fixture()), { search: '?run_id=initial-run' });
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(app.nodes.get('detail').open, true);
  app.nodes.get('detail').dispatch('cancel');
  const initialCalls = app.calls.length;
  for (const hash of ['#workflows', '#explorer', '#template-section', '']) {
    app.sandbox.location.hash = hash;
    app.dispatchWindow('hashchange');
    assert.equal(app.nodes.get('detail').open, false, hash + ' must not reopen detail');
    assert.equal(app.calls.length, initialCalls, hash + ' must not fetch the query run again');
  }
  app.sandbox.location.hash = '#run=another-run';
  await app.dispatchWindow('hashchange');
  assert.equal(app.nodes.get('detail').open, true);
  assert.equal(app.nodes.get('detail-title').textContent, 'Run another-run');
  assert.equal(app.calls.filter(x => x.url === '/api/benchmarks/runs/initial-run').length, 1);
  assert.equal(app.calls.filter(x => x.url === '/api/benchmarks/runs/another-run').length, 1);
});

test('analytics never fan out over registry pages and run controls are keyboard-native', async () => {
  const app = browser();
  await app.sandbox.load();
  const queries = app.calls.filter(x => x.url.startsWith('/api/benchmarks/runs?'));
  assert.equal(queries.length, 2); // Initial load and this explicit refresh.
  assert(queries.every(x => new URL('http://synthetic' + x.url).searchParams.get('per_page') === '25'));
  assert.match(app.nodes.get('analytics-scope').textContent, /current page only/);
  app.sandbox.renderTable([{ run_id: 'keyboard-run', status: 'planned' }]);
  const open = walk(app.nodes.get('runs')).find(x => x.tagName === 'BUTTON');
  assert.equal(open.attributes['aria-label'], 'Open run keyboard-run');
  assert(html.includes('aria-labelledby="detail-title"'));
  assert(html.includes('scope="col"'));
});
