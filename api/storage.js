// api/storage.js — proxy per Upstash Redis REST
// Operacions suportades: GET, SET, DEL, KEYS
export const config = {
  api: { bodyParser: { sizeLimit: "16mb" } },
};

const upstash = async (cmd, ...args) => {
  const url  = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;
  if (!url || !token) throw new Error("Upstash no configurat");

  const r = await fetch(`${url}/${[cmd, ...args.map(a => encodeURIComponent(JSON.stringify(a)))].join("/")}`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  const d = await r.json();
  if (d.error) throw new Error(d.error);
  return d.result;
};

export default async function handler(req, res) {
  const { op, key, value, pattern } = req.method === "POST"
    ? (req.body || {})
    : req.query;

  try {
    switch (op) {
      case "get": {
        const val = await upstash("get", key);
        return res.json({ value: val });
      }
      case "set": {
        await upstash("set", key, value);
        return res.json({ ok: true });
      }
      case "del": {
        await upstash("del", key);
        return res.json({ ok: true });
      }
      case "keys": {
        const keys = await upstash("keys", pattern || "*");
        return res.json({ keys: keys || [] });
      }
      default:
        return res.status(400).json({ error: `Op desconeguda: ${op}` });
    }
  } catch (e) {
    return res.status(500).json({ error: e.message });
  }
}