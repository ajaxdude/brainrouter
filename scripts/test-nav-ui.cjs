#!/usr/bin/env node
'use strict';

const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const html = fs.readFileSync(path.join(__dirname, '../src/escalation/templates/main_dashboard.html'), 'utf8');

function ids() { return [...html.matchAll(/\bid="([^"]+)"/g)].map(m => m[1]); }
function count(re) { return (html.match(re) || []).length; }
function requireOneId(id) { assert.equal(ids().filter(x => x === id).length, 1, `expected one #${id}`); }
function tagForId(id) {
  const re = new RegExp(`<([a-zA-Z][\\w-]*)\\b[^>]*\\bid="${id.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}"[^>]*>`, 'm');
  const m = html.match(re);
  assert.ok(m, `missing #${id}`);
  return m[0];
}
function assertOnclick(id, expected) { assert.ok(tagForId(id).includes(`onclick="${expected}"`), `#${id} lost onclick ${expected}`); }
function extractFn(name) {
  const start = html.indexOf(`function ${name}(`);
  assert.notEqual(start, -1, `missing function ${name}`);
  let brace = html.indexOf('{', start), depth = 0;
  for (let i = brace; i < html.length; i++) {
    if (html[i] === '{') depth++;
    if (html[i] === '}') {
      depth--;
      if (depth === 0) return html.slice(start, i + 1);
    }
  }
  throw new Error(`unterminated ${name}`);
}

class Element {
  constructor(id) {
    this.id = id; this.style = {}; this.attrs = {}; this.textContent = ''; this.innerHTML = ''; this.value = ''; this.disabled = false;
    this.classList = { values: new Set(), toggle: (n, on) => { if (on) this.classList.values.add(n); else this.classList.values.delete(n); }, contains: n => this.classList.values.has(n) };
  }
  setAttribute(k,v){ this.attrs[k] = String(v); }
  removeAttribute(k){ delete this.attrs[k]; }
}
function sandboxFor(functionNames, extra = {}) {
  const elements = new Map();
  for (const id of ids()) if (!id.includes('${')) elements.set(id, new Element(id));
  const sandbox = {
    document: { getElementById: id => elements.get(id) || null },
    healthData: {}, nudgeData: null, codeReviewEnabled: null, prGuidelinesEnabled: null, hankndoryEnabled: null,
    activeView: 'stream', loadConfig(){}, loadModels(){}, loadToolboxModels(){ sandbox.downloadGate = (sandbox.downloadGate || 0) + 1; }, loadServerMode(){}, escHtml: s => String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'),
    ...extra,
  };
  vm.createContext(sandbox);
  vm.runInContext(functionNames.map(extractFn).join('\n'), sandbox, { filename: 'nav-ui.js' });
  sandbox.__elements = elements;
  return sandbox;
}

test('primary nav has four destinations plus Models subtabs and external Model activity', () => {
  for (const id of ['nav-stream','nav-models','nav-benchmarks','nav-config','models-subnav','models-tab-local','models-tab-downloads','models-tab-serving','models-activity-link']) requireOneId(id);
  assertOnclick('nav-stream', "switchView('stream')");
  assertOnclick('nav-models', "switchView('models')");
  assertOnclick('nav-config', "switchView('config')");
  assert.ok(tagForId('nav-benchmarks').includes('href="/benchmarks"'));
  assert.ok(tagForId('models-activity-link').includes('href="/models"'));
  assertOnclick('models-tab-local', "switchView('models')");
  assertOnclick('models-tab-downloads', "switchView('toolbox-models')");
  assertOnclick('models-tab-serving', "switchView('server-mode')");
  assert.equal(count(/id="nav-toolbox-models"/g), 0, 'Downloads must not remain a primary nav item');
  assert.equal(count(/id="nav-server-mode"/g), 0, 'Server Mode must not remain a primary nav item');
});

test('relocated controls retain ids and onclick handlers', () => {
  for (const id of ['routing-preset','main-backend','main-model','reviewer-backend','reviewer-model','reviewer-model-select','subagent-model-select','save-routing-profile','routing-profile-message','routing-model-errors','routing-saved','routing-last-actual','routing-hint']) requireOneId(id);
  assertOnclick('save-routing-profile', 'saveRoutingProfile()');
  assertOnclick('toggle-nudge', 'toggleNudge()');
  assertOnclick('tier-auto', "setNudgeTier('auto')");
  assertOnclick('tier-light', "setNudgeTier('light')");
  assertOnclick('tier-deep', "setNudgeTier('deep')");
  assertOnclick('toggle-codereview', 'toggleCodeReview()');
  assertOnclick('toggle-prguidelines', 'togglePrGuidelines()');
  assertOnclick('toggle-hankndory', 'toggleHankndory()');
  assertOnclick('toggle-rewrite', 'togglePromptRewrite()');
  assertOnclick('toggle-discord', "toggleBridge('discord')");
  assertOnclick('toggle-signal', "toggleBridge('signal')");
  for (const id of ['toolbox-list','cockpit-config-status','service-controls','quality-settings','integrations-settings','toolbox-management','review-ledger','review-hankndory-state']) requireOneId(id);
  for (const svc of ['llama-swap','llama-cpp','manifest','brainrouter']) assert.ok(html.includes(`restartService('${svc}')`), `${svc} restart missing`);
  assert.ok(html.includes('onclick="syncModels()"'));
  assert.ok(html.includes('onclick="flushModels()"'));
  assert.ok(html.includes('upgradeToolboxAll()'));
});

test('no duplicate concrete ids', () => {
  const seen = new Map();
  for (const id of ids().filter(x => !x.includes('${'))) seen.set(id, (seen.get(id) || 0) + 1);
  const dup = [...seen].filter(([, n]) => n > 1);
  assert.deepEqual(dup, []);
});

test('switchView keeps Models parent active across subtabs and fires downloads gate', () => {
  const s = sandboxFor(['isModelsView','switchView']);
  s.switchView('models');
  assert.equal(s.__elements.get('nav-models').classList.contains('active'), true);
  assert.equal(s.__elements.get('models-tab-local').attrs['aria-selected'], 'true');
  s.switchView('toolbox-models');
  assert.equal(s.activeView, 'toolbox-models');
  assert.equal(s.downloadGate, 1);
  assert.equal(s.__elements.get('nav-models').classList.contains('active'), true);
  assert.equal(s.__elements.get('models-tab-downloads').attrs['aria-selected'], 'true');
  s.switchView('server-mode');
  assert.equal(s.__elements.get('nav-models').classList.contains('active'), true);
  assert.equal(s.__elements.get('models-tab-serving').attrs['aria-selected'], 'true');
});

test('renderPosture maps existing sources and loading states', () => {
  const s = sandboxFor(['postureState','renderPosture']);
  assert.equal(s.renderPosture().text, 'reasoning loading · review loading · HankNDory loading · PR loading');
  s.nudgeData = { enabled: true, tier: 'deep' };
  s.codeReviewEnabled = false;
  s.hankndoryEnabled = true;
  s.prGuidelinesEnabled = true;
  const out = s.renderPosture();
  assert.deepEqual({ reasoning: out.reasoning, review: out.review, hankndory: out.hankndory, pr: out.pr }, { reasoning: 'deep', review: 'off', hankndory: 'on', pr: 'on' });
  assert.match(s.__elements.get('posture-line').innerHTML, /#quality-settings/);
  assert.match(s.__elements.get('posture-line').innerHTML, /#review-ledger-panel/);
  s.nudgeData = { enabled: false, tier: 'light' };
  assert.equal(s.renderPosture().reasoning, 'off');
});

test('renderHealthPill colors follow precedence and ignore Bonsai', () => {
  const s = sandboxFor(['healthPillState','renderHealthPill']);
  assert.equal(s.renderHealthPill().color, 'grey');
  s.healthData = { llama_swap: 'down', llama_cpp: 'healthy', manifest: 'healthy', toolbox: 'healthy' };
  assert.equal(s.renderHealthPill().color, 'red');
  s.healthData = { llama_swap: 'healthy', llama_cpp: 'healthy', manifest: 'healthy', toolbox: 'loading' };
  assert.equal(s.renderHealthPill().color, 'amber');
  s.healthData = { llama_swap: 'healthy', llama_cpp: 'healthy', manifest: 'disabled', toolbox: 'healthy', cloud_fallback: false };
  assert.equal(s.renderHealthPill().color, 'amber');
  s.healthData = { llama_swap: 'healthy', llama_cpp: 'healthy', manifest: 'healthy', toolbox: 'healthy', cloud_fallback: true, bonsai: 'down' };
  assert.equal(s.renderHealthPill().color, 'amber');
  s.healthData = { llama_swap: 'healthy', llama_cpp: 'healthy', manifest: 'healthy', toolbox: 'healthy', cloud_fallback: false, bonsai: 'down' };
  assert.equal(s.renderHealthPill().color, 'green');
});

test('gufo remains in Downloads and Server Mode backend tabs', () => {
  assert.match(html, /const DOWNLOAD_CAPABLE_BACKENDS = \[[^\]]*'gufo'/);
  assert.match(html, /const SERVER_MODE_CAPABLE_BACKENDS = \[[^\]]*'gufo'/);
  assert.ok(html.includes('sm-panel-gufo'));
  assert.ok(html.includes("backend === 'gufo'"));
});
