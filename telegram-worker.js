export default {
  async fetch(request, env) {
    const cors = {
      'Access-Control-Allow-Origin': env.ALLOWED_ORIGIN || '*',
      'Access-Control-Allow-Headers': 'content-type,x-client-key',
      'Access-Control-Allow-Methods': 'POST,OPTIONS',
      'Content-Type': 'application/json; charset=utf-8'
    };
    if (request.method === 'OPTIONS') return new Response(null, { headers: cors });
    if (request.method !== 'POST') return json({ ok:false, error:'POST only' }, 405, cors);
    const url = new URL(request.url);
    if (url.pathname !== '/signal') return json({ ok:false, error:'Not found' }, 404, cors);
    if (!env.TELEGRAM_BOT_TOKEN || !env.TELEGRAM_CHAT_ID || !env.CLIENT_KEY) return json({ ok:false, error:'Worker secrets missing' }, 500, cors);
    if (request.headers.get('x-client-key') !== env.CLIENT_KEY) return json({ ok:false, error:'Unauthorized' }, 401, cors);
    let body;
    try { body = await request.json(); } catch { return json({ ok:false, error:'Invalid JSON' }, 400, cors); }
    const allowed = new Set(['BUY','SELL','WAIT','TEST']);
    const action = String(body.action || '').toUpperCase();
    if (!allowed.has(action)) return json({ ok:false, error:'Invalid action' }, 400, cors);
    const text = [
      `📊 369 PAPER SIGNAL: ${action}`,
      `Symbol: ${body.symbol || 'PAXGUSDT'}`,
      `Price: ${body.price ?? '—'}`,
      `Entry: ${body.entry ?? '—'}`,
      `SL: ${body.sl ?? '—'}`,
      `TP1: ${body.tp1 ?? '—'}`,
      `TP2: ${body.tp2 ?? '—'}`,
      `Signal engine: ${body.signal || '—'}`,
      `Source: ${body.source || 'PAXG proxy paper only'}`,
      `Time: ${body.ts || new Date().toISOString()}`,
      '',
      '⚠️ Paper trade only. Không phải lệnh MT5/GC/MGC.'
    ].join('\n');
    const tg = await fetch(`https://api.telegram.org/bot${env.TELEGRAM_BOT_TOKEN}/sendMessage`, {
      method:'POST', headers:{'content-type':'application/json'},
      body:JSON.stringify({ chat_id:env.TELEGRAM_CHAT_ID, text })
    });
    const result = await tg.json();
    if (!tg.ok || !result.ok) return json({ ok:false, error:result.description || 'Telegram error' }, 502, cors);
    return json({ ok:true }, 200, cors);
  }
};
function json(data, status, headers) { return new Response(JSON.stringify(data), { status, headers }); }
