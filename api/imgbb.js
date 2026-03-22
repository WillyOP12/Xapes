export const config = {
  api: { bodyParser: { sizeLimit: "12mb" } },
};

export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const key = process.env.imgbb_api;
  if (!key) return res.status(500).json({ error: "imgbb_api no configurada al servidor" });

  let body = req.body;
  if (typeof body === "string") {
    try { body = JSON.parse(body); } catch { return res.status(400).json({ error: "JSON invàlid" }); }
  }

  const { image, name } = body || {};
  if (!image) return res.status(400).json({ error: "Falta el camp image" });

  try {
    // URLSearchParams = application/x-www-form-urlencoded, suportat per ImgBB
    const fd = new URLSearchParams();
    fd.append("image", image);
    if (name) fd.append("name", String(name));

    const response = await fetch(`https://api.imgbb.com/1/upload?key=${key}`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: fd.toString(),
    });

    const text = await response.text();
    let data;
    try { data = JSON.parse(text); }
    catch { return res.status(502).json({ error: `ImgBB resposta no-JSON (HTTP ${response.status}): ${text.slice(0, 300)}` }); }

    res.status(response.status).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}