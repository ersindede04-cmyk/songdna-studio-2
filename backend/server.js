
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");
const crypto = require("node:crypto");

// Minimal .env loader
try {
  const envFile = fs.readFileSync(path.join(__dirname, ".env"), "utf8");
  for (const line of envFile.split(/\r?\n/)) {
    const match = line.match(/^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$/);
    if (match && process.env[match[1]] === undefined) {
      process.env[match[1]] = match[2].replace(/^['"]|['"]$/g, "");
    }
  }
} catch (_) {}

const PORT = Number(process.env.PORT || 8080);
const APP_ACCESS_TOKEN = process.env.APP_ACCESS_TOKEN || "";
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const GEMINI_API_KEY = process.env.GEMINI_API_KEY || "";
const OPENAI_MODEL = process.env.OPENAI_MODEL || "gpt-4.1-mini";
const GEMINI_MODEL = process.env.GEMINI_MODEL || "gemini-2.5-flash";
const ALLOWED_ORIGIN = process.env.ALLOWED_ORIGIN || "*";
const TRUST_PROXY = process.env.TRUST_PROXY === "true";

const WINDOW_MS = 60_000;
const MAX_REQUESTS = 12;
const MAX_BODY = 40_000;
const AUTH_WINDOW_MS = 5 * 60_000;
const MAX_AUTH_FAILURES = 8;

const buckets = new Map();
const authFailures = new Map();

const OPENAI_MODELS = (process.env.OPENAI_MODELS || OPENAI_MODEL)
  .split(",").map(x => x.trim());

const GEMINI_MODELS = (process.env.GEMINI_MODELS || GEMINI_MODEL)
  .split(",").map(x => x.trim());

if (!APP_ACCESS_TOKEN || APP_ACCESS_TOKEN.length < 12) {
  console.error("APP_ACCESS_TOKEN zorunlu (en az 12 karakter). Sunucu başlatılmadı.");
  process.exit(1);
}

const tokenOk = t => {
  const a = Buffer.from(String(t || ""));
  const b = Buffer.from(APP_ACCESS_TOKEN);
  return a.length === b.length && crypto.timingSafeEqual(a, b);
};

const clientIp = req => {
  if (TRUST_PROXY) {
    const forwarded = String(req.headers["x-forwarded-for"] || "").split(",")[0].trim();
    if (forwarded) return forwarded;
  }
  return req.socket.remoteAddress || "unknown";
};

setInterval(() => {
  const now = Date.now();
  for (const [k, v] of buckets) {
    if (!v.some(t => now - t < WINDOW_MS)) buckets.delete(k);
  }
  for (const [k, v] of authFailures) {
    if (!v.some(t => now - t < AUTH_WINDOW_MS)) authFailures.delete(k);
  }
}, 300_000).unref();

const pick = (list, m, def) => (m && list.includes(m)) ? m : def;

function send(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
    "Access-Control-Allow-Headers": "Content-Type, Authorization, X-App-Token",
    "Cache-Control": "no-store"
  });
  res.end(body);
}

function rateLimited(ip) {
  const now = Date.now();
  const old = buckets.get(ip) || [];
  const recent = old.filter(t => now - t < WINDOW_MS);
  if (recent.length >= MAX_REQUESTS) return true;
  recent.push(now);
  buckets.set(ip, recent);
  return false;
}

function authLocked(ip) {
  const now = Date.now();
  const recent = (authFailures.get(ip) || []).filter(t => now - t < AUTH_WINDOW_MS);
  authFailures.set(ip, recent);
  return recent.length >= MAX_AUTH_FAILURES;
}

function recordAuthFailure(ip) {
  const now = Date.now();
  const recent = (authFailures.get(ip) || []).filter(t => now - t < AUTH_WINDOW_MS);
  recent.push(now);
  authFailures.set(ip, recent);
}

async function readBody(req) {
  let body = "";
  for await (const chunk of req) {
    body += chunk;
    if (body.length > MAX_BODY) throw new Error("İstek çok büyük.");
  }
  return JSON.parse(body || "{}");
}

async function openai(prompt, model) {
  if (!OPENAI_API_KEY) {
    throw new Error("Sunucuda OPENAI_API_KEY tanımlı değil.");
  }

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    signal: AbortSignal.timeout(60000),
    headers: {
      "Authorization": `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      model: pick(OPENAI_MODELS, model, OPENAI_MODEL),
      temperature: 0.85,
      messages: [
        {
          role: "system",
          content: "You are a professional songwriter. Follow the user's creative brief exactly. Return only a short TITLE line and complete original song lyrics. Never return placeholders or instructions."
        },
        { role: "user", content: prompt }
      ]
    })
  });

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error?.message || `OpenAI isteği başarısız (${response.status}).`);
  }
  return data.choices?.[0]?.message?.content || "";
}

async function gemini(prompt, model) {
  if (!GEMINI_API_KEY) {
    throw new Error("Sunucuda GEMINI_API_KEY tanımlı değil.");
  }

  const selected = pick(GEMINI_MODELS, model, GEMINI_MODEL);
  const response = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(selected)}:generateContent`,
    {
      method: "POST",
      signal: AbortSignal.timeout(60000),
      headers: {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
      },
      body: JSON.stringify({
        systemInstruction: {
          parts: [{
            text: "You are a professional songwriter. Follow the user's creative brief exactly. Return only a short TITLE line and complete original song lyrics. Never return placeholders or instructions."
          }]
        },
        contents: [{ role: "user", parts: [{ text: prompt }] }],
        generationConfig: { temperature: 0.85 }
      })
    }
  );

  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error?.message || `Gemini isteği başarısız (${response.status}).`);
  }

  return (data.candidates?.[0]?.content?.parts || [])
    .map(p => p.text || "").join("\n");
}

const server = http.createServer(async (req, res) => {
  if (req.method === "OPTIONS") {
    res.writeHead(204, {
      "Access-Control-Allow-Origin": ALLOWED_ORIGIN,
      "Access-Control-Allow-Methods": "POST, OPTIONS, GET",
      "Access-Control-Allow-Headers": "Content-Type, Authorization, X-App-Token",
      "Access-Control-Max-Age": "86400"
    });
    return res.end();
  }

  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

  if (req.method === "GET" && url.pathname === "/health") {
    return send(res, 200, {
      ok: true,
      service: "SONGDNA AI",
      providers: {
        openai: !!OPENAI_API_KEY,
        gemini: !!GEMINI_API_KEY
      }
    });
  }

  if (req.method !== "POST" || url.pathname !== "/api/lyrics") {
    return send(res, 404, { error: "Endpoint bulunamadı." });
  }

  const ip = clientIp(req);

  if (authLocked(ip)) {
    return send(res, 429, {
      error: "Çok fazla başarısız erişim kodu denemesi. Beş dakika sonra tekrar dene."
    });
  }

  if (!tokenOk(req.headers["x-app-token"])) {
    recordAuthFailure(ip);
    return send(res, 401, { error: "Uygulama erişim kodu geçersiz." });
  }

  authFailures.delete(ip);

  if (rateLimited(ip)) {
    return send(res, 429, {
      error: "Çok fazla istek. Bir dakika sonra tekrar dene."
    });
  }

  try {
    let input;

    try {
      input = await readBody(req);
    } catch (error) {
      if (error instanceof SyntaxError) {
        return send(res, 400, { error: "İstek gövdesi geçerli JSON olmalı." });
      }
      if (error.message === "İstek çok büyük.") {
        return send(res, 413, { error: error.message });
      }
      throw error;
    }

    const prompt = typeof input.prompt === "string" ? input.prompt.trim() : "";

    if (!prompt || prompt.length > 20_000) {
      return send(res, 400, {
        error: "Geçerli bir söz üretim istemi gerekli (en fazla 20.000 karakter)."
      });
    }

    if (input.provider !== undefined && !["openai", "gemini"].includes(input.provider)) {
      return send(res, 400, { error: "Desteklenmeyen AI sağlayıcısı." });
    }

    const provider = input.provider === "gemini" ? "gemini" : "openai";

    if (provider === "openai" && !OPENAI_API_KEY) {
      return send(res, 503, {
        error: "OpenAI sağlayıcısı sunucuda yapılandırılmamış."
      });
    }

    if (provider === "gemini" && !GEMINI_API_KEY) {
      return send(res, 503, {
        error: "Gemini sağlayıcısı sunucuda yapılandırılmamış."
      });
    }

    let lyrics;

    try {
      lyrics = provider === "gemini"
        ? await gemini(prompt, input.model)
        : await openai(prompt, input.model);
    } catch (providerError) {
      console.error("AI provider error:", providerError);
      const timedOut =
        providerError.name === "AbortError" ||
        providerError.name === "TimeoutError";

      return send(res, timedOut ? 504 : 502, {
        error: timedOut
          ? "AI sağlayıcısı zamanında yanıt vermedi."
          : "AI sağlayıcısından yanıt alınamadı."
      });
    }

    if (!lyrics.trim()) {
      return send(res, 502, { error: "Yapay zekâ boş yanıt verdi." });
    }

    return send(res, 200, {
      provider,
      model: provider === "gemini"
        ? pick(GEMINI_MODELS, input.model, GEMINI_MODEL)
        : pick(OPENAI_MODELS, input.model, OPENAI_MODEL),
      lyrics
    });

  } catch (error) {
    console.error(error);
    return send(res, 500, {
      error: "Sunucuda beklenmeyen bir hata oluştu."
    });
  }
});

server.listen(PORT, "0.0.0.0", () => {
  console.log(`SONGDNA AI server listening on ${PORT}`);
});
      
