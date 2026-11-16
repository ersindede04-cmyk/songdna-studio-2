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
  .split(',')
  .map(x => x.trim())
  .filter(x => ALLOWED_FLASH_MODELS.has(x));

const GEMINI_MODELS = [...new Set(configuredModels)].slice(0, 3);

if (!GEMINI_MODELS.length) GEMINI_MODELS.push(DEFAULT_MODEL);

const requestedDefault = process.env.GEMINI_MODEL || DEFAULT_MODEL;

const GEMINI_MODEL = GEMINI_MODELS.includes(requestedDefault)
  ? requestedDefault
  : GEMINI_MODELS[0];

const TOTAL_GENERATION_BUDGET_MS = 78_000;
const PER_MODEL_TIMEOUT_MS = 24_000;

const WINDOW_MS = 60_000;
const MAX_REQUESTS = 12;
const MAX_BODY = 40_000;
const AUTH_WINDOW_MS = 5 * 60_000;
const MAX_AUTH_FAILURES = 8;

const buckets = new Map();
const authFailures = new Map();

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
    const forwarded = String(req.headers['x-forwarded-for'] || '')
      .split(',')[0]
      .trim();

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

  const recent = (buckets.get(ip) || [])
    .filter(t => now - t < WINDOW_MS);

  if (recent.length >= MAX_REQUESTS) return true;

  recent.push(now);
  buckets.set(ip, recent);

  return false;
}

function authLocked(ip) {
  const now = Date.now();

  const recent = (authFailures.get(ip) || [])
    .filter(t => now - t < AUTH_WINDOW_MS);

  authFailures.set(ip, recent);

  return recent.length >= MAX_AUTH_FAILURES;
}

function recordAuthFailure(ip) {
  const now = Date.now();

  const recent = (authFailures.get(ip) || [])
    .filter(t => now - t < AUTH_WINDOW_MS);

  recent.push(now);
  authFailures.set(ip, recent);
}

async function readBody(req) {
  let body = '';

  for await (const chunk of req) {
    body += chunk;

    if (body.length > MAX_BODY) {
      throw Object.assign(
        new Error('İstek çok büyük.'),
        { status: 413 }
      );
    }
  }

  try {
    return JSON.parse(body || '{}');
  } catch (_) {
    throw Object.assign(
      new Error('İstek gövdesi geçerli JSON olmalı.'),
      { status: 400 }
    );
  }
}

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

function responseText(data) {
  return (data.candidates?.[0]?.content?.parts || [])
    .map(p => p.text || '')
    .join('\n')
    .trim();
}

async function callGemini(model, prompt, task, timeoutMs, requestSignal) {
  const systemText = task === 'style'
    ? 'You are an expert Suno music-style prompt designer. Return only one concise, production-ready STYLE prompt in English. Preserve the user\'s requested genre blend, instruments, BPM, energy, vocal character and exclusions. Never write lyrics. Maximum 1000 characters.'
    : 'You are a professional songwriter. Follow the creative brief exactly. Return a short TITLE line and complete original song lyrics. Use clear Suno section labels in English square brackets. Do not return placeholders, explanations or production notes.';

  let response;

  try {
    response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`,
      {
        method: 'POST',
        signal: requestSignal
          ? AbortSignal.any([
              AbortSignal.timeout(timeoutMs),
              requestSignal
            ])
          : AbortSignal.timeout(timeoutMs),

        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': GEMINI_API_KEY
        },

        body: JSON.stringify({
          systemInstruction: {
            parts: [{ text: systemText }]
          },

          contents: [
            {
              role: 'user',
              parts: [{ text: prompt }]
            }
          ],

          generationConfig: {}
        })
      }
    );
  } catch (cause) {
    const timedOut = cause && (
      cause.name === 'TimeoutError' ||
      cause.name === 'AbortError'
    );

    throw Object.assign(
      new Error(
        timedOut
          ? 'Gemini isteği zaman aşımına uğradı.'
          : 'Gemini sunucusuna bağlanılamadı.'
      ),
      {
        status: timedOut ? 504 : 502,
        retryable: true,
        cause
      }
    );
  }

  let data;

  try {
    data = await response.json();
  } catch (_) {
    throw Object.assign(
      new Error('Google Gemini geçerli JSON yanıtı vermedi.'),
      {
        status: response.status || 502,
        retryable: response.status >= 500
      }
    );
  }

  if (!response.ok) {
    const message = data.error?.message ||
      `Gemini isteği başarısız (${response.status}).`;

    const error = new Error(message);

    error.status = response.status;

    error.retryable = [
      408,
      429,
      500,
      502,
      503,
      504
    ].includes(response.status);

    throw error;
  }

  const text = responseText(data);

  if (!text) {
    throw Object.assign(
      new Error('Gemini boş yanıt verdi.'),
      {
        status: 502,
        retryable: true
      }
    );
  }

  return text;
}

async function generateWithFallback(
  prompt,
  requestedModel,
  task,
  requestSignal
) {
  const first = GEMINI_MODELS.includes(requestedModel)
    ? requestedModel
    : GEMINI_MODEL;

  const order = [
    first,
    ...GEMINI_MODELS.filter(m => m !== first)
  ];

  const deadline = Date.now() + TOTAL_GENERATION_BUDGET_MS;
  const failures = [];

  for (const model of order) {
    const remaining = deadline - Date.now();

    if (remaining < 1500) break;

    try {
      const text = await callGemini(
        model,
        prompt,
        task,
        Math.min(PER_MODEL_TIMEOUT_MS, remaining - 500),
        requestSignal
      );

      return {
        text,
        model,
        fallbackUsed: model !== first
      };
    } catch (error) {
      failures.push({
        model,
        status: error.status || 0,
        message: error.message
      });

      console.error(
        `Gemini ${task} error [${model}]:`,
        error.message
      );

      if (requestSignal?.aborted) throw error;

      if (
        !error.retryable &&
        ![429, 503].includes(error.status)
      ) {
        break;
      }
    }
  }

  const last = failures[failures.length - 1] || {
    status: 504,
    message: 'Zaman bütçesi doldu'
  };

  const status = last.status === 429
    ? 429
    : last.status === 503
      ? 503
      : [408, 504].includes(last.status)
        ? 504
        : 502;

  const publicMessage = status === 429
    ? 'Gemini kullanım kotasına ulaşıldı. Kota yenilenince tekrar deneyebilirsin.'
    : status === 503
      ? 'Gemini şu anda yanıt vermiyor. Biraz sonra tekrar dene.'
      : status === 504
        ? 'Gemini yanıt süresi doldu. Yedek Flash modeller de zamanında yanıt vermedi; tekrar dene.'
        : 'Gemini üretimi başarısız oldu. Model ayarlarını ve sunucu kayıtlarını kontrol et.';

  throw Object.assign(
    new Error(publicMessage),
    { status }
  );
}

setInterval(() => {
  const now = Date.now();

  for (const [key, values] of buckets) {
    if (!values.some(t => now - t < WINDOW_MS)) {
      buckets.delete(key);
    }
  }

  for (const [key, values] of authFailures) {
    if (!values.some(t => now - t < AUTH_WINDOW_MS)) {
      authFailures.delete(key);
    }
  }
}, 300_000).unref();

const server = http.createServer(async (req, res) => {
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': ALLOWED_ORIGIN,
      'Access-Control-Allow-Methods': 'POST, OPTIONS, GET',
      'Access-Control-Allow-Headers': 'Content-Type, X-App-Token',
      'Access-Control-Max-Age': '86400'
    });

    return res.end();
  }

  const url = new URL(
    req.url,
    `http://${req.headers.host || 'localhost'}`
  );

  if (req.method === 'GET' && url.pathname === '/health') {
    return send(res, 200, {
      ok: true,
      service: 'SONGDNA Studio 4.0.2 AI',

      providers: {
        gemini: !!GEMINI_API_KEY
      },

      models: GEMINI_MODELS,
      defaultModel: GEMINI_MODEL,

      billingPolicy: 'flash-model-allowlist-only; no premium-model fallback; Google project billing tier applies'
    });
  }

  if (
    req.method !== 'POST' ||
    !['/api/lyrics', '/api/style'].includes(url.pathname)
  ) {
    return send(res, 404, {
      error: 'Endpoint bulunamadı.'
    });
  }

  const ip = clientIp(req);

  if (authLocked(ip)) {
    return send(res, 429, {
      error: 'Çok fazla başarısız erişim kodu denemesi. Beş dakika sonra tekrar dene.'
    });
  }

  if (!tokenOk(req.headers['x-app-token'])) {
    recordAuthFailure(ip);

    return send(res, 401, {
      error: 'Uygulama erişim kodu geçersiz.'
    });
  }

  authFailures.delete(ip);

  if (rateLimited(ip)) {
    return send(res, 429, {
      error: 'Çok fazla istek. Bir dakika sonra tekrar dene.'
    });
  }

  if (!GEMINI_API_KEY) {
    return send(res, 503, {
      error: 'Sunucuda GEMINI_API_KEY tanımlı değil.'
    });
  }

  try {
    const input = await readBody(req);

    const prompt = typeof input.prompt === 'string'
      ? input.prompt.trim()
      : '';

    if (!prompt || prompt.length > 20_000) {
      return send(res, 400, {
        error: 'Geçerli bir istem gerekli (en fazla 20.000 karakter).'
      });
    }

    const task = url.pathname === '/api/style'
      ? 'style'
      : 'lyrics';

    const disconnectController = new AbortController();

    res.on('close', () => {
      if (!res.writableEnded) {
        disconnectController.abort();
      }
    });

    const result = await generateWithFallback(
      prompt,
      input.model,
      task,
      disconnectController.signal
    );

    return send(
      res,
      200,
      task === 'style'
        ? {
            provider: 'gemini',
            model: result.model,
            fallbackUsed: result.fallbackUsed,
            style: result.text
          }
        : {
            provider: 'gemini',
            model: result.model,
            fallbackUsed: result.fallbackUsed,
            lyrics: result.text
          }
    );

  } catch (error) {
    if (res.destroyed || res.writableEnded) return;

    if (error.name !== 'AbortError') {
      console.error('Request failed:', error.message);
    }

    const status = error.status || 500;

    return send(res, status, {
      error: error.message ||
        'Sunucuda beklenmeyen bir hata oluştu.'
    });
  }
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(
    `SONGDNA Studio 4.0.2 Gemini server listening on ${PORT}`
  );
});
