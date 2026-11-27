```javascript
'use strict';

const http = require('node:http');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

// Optional local .env loader. Production secrets belong in the hosting provider's environment.
try {
  const envText = fs.readFileSync(path.join(__dirname, '.env'), 'utf8');
  for (const line of envText.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (match && process.env[match[1]] === undefined) {
      process.env[match[1]] = match[2].replace(/^['"]|['"]$/g, '');
    }
  }
} catch (_) {}

const PORT = Number(process.env.PORT || 8080);
const APP_ACCESS_TOKEN = process.env.APP_ACCESS_TOKEN || '';
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || '';
const GEMINI_API_KEY_2 = process.env.GEMINI_API_KEY_2 || '';
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || '*';
const TRUST_PROXY = process.env.TRUST_PROXY === 'true';
// Flash text-model allowlist only. Pro/image/video/audio models are intentionally excluded; billing tier is controlled by the Google Cloud project, not the model ID.
const ALLOWED_FLASH_MODELS = new Set([
  'gemini-3.1-flash-lite',
  'gemini-3.6-flash',
  'gemini-3.8-flash'
]);
const DEFAULT_MODEL = 'gemini-3.1-flash-lite';
const configuredModels = (process.env.GEMINI_MODELS || DEFAULT_MODEL)
  .split(',').map(x => x.trim()).filter(x => ALLOWED_FLASH_MODELS.has(x));
const GEMINI_MODELS = [...new Set(configuredModels)].slice(0, 3);
if (!GEMINI_MODELS.length) GEMINI_MODELS.push(DEFAULT_MODEL);
const requestedDefault = process.env.GEMINI_MODEL || DEFAULT_MODEL;
const GEMINI_MODEL = GEMINI_MODELS.includes(requestedDefault) ? requestedDefault : GEMINI_MODELS[0];
const TOTAL_GENERATION_BUDGET_MS = 78_000;
// Story blending uses one Gemini generation request (with the existing model fallback policy).
// This keeps the optional feature comfortably inside the Android client's 150s timeout.
const PER_MODEL_TIMEOUT_MS = 24_000;

const WINDOW_MS = 60_000;
const MAX_REQUESTS = 12;
// 150,000 JS characters can occupy substantially more UTF-8 bytes in JSON. Keep a safe transport ceiling.
const MAX_BODY = 1_000_000;
const buckets = new Map();

if (!APP_ACCESS_TOKEN || APP_ACCESS_TOKEN.length < 12) {
  console.error('APP_ACCESS_TOKEN zorunlu (en az 12 karakter). Sunucu başlatılmadı.');
  process.exit(1);
}

function tokenOk(token) {
  const a = Buffer.from(String(token || ''));
  const b = Buffer.from(APP_ACCESS_TOKEN);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
}
function clientIp(req) {
  if (TRUST_PROXY) {
    const forwarded = String(req.headers['x-forwarded-for'] || '').split(',')[0].trim();
    if (forwarded) return forwarded;
  }
  return req.socket.remoteAddress || 'unknown';
}
function send(res, status, data) {
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
    'Access-Control-Allow-Methods': 'POST, OPTIONS, GET',
    'Access-Control-Allow-Headers': 'Content-Type, X-App-Token',
    'Cache-Control': 'no-store'
  });
  res.end(JSON.stringify(data));
}
function rateLimited(ip) {
  const now = Date.now();
  const recent = (buckets.get(ip) || []).filter(t => now - t < WINDOW_MS);
  if (recent.length >= MAX_REQUESTS) return true;
  recent.push(now);
  buckets.set(ip, recent);
  return false;
}
async function readBody(req) {
  let body = '';
  for await (const chunk of req) {
    body += chunk;
    if (body.length > MAX_BODY) throw Object.assign(new Error('İstek çok büyük.'), { status: 413 });
  }
  try { return JSON.parse(body || '{}'); }
  catch (_) { throw Object.assign(new Error('İstek gövdesi geçerli JSON olmalı.'), { status: 400 }); }
}
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function responseText(data) {
  return (data.candidates?.[0]?.content?.parts || []).map(p => p.text || '').join('\n').trim();
}

// The command endpoint is a structured selector parser, not a free-form content endpoint.
