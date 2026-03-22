export default async function handler(req, res) {
  if (req.method !== "POST") return res.status(405).json({ error: "Method not allowed" });

  const key = process.env.imgbb_api;
  if (!key) return res.status(500).json({ error: "imgbb_api not configured" });

  const { image, name } = req.body;
  if (!image) return res.status(400).json({ error: "Missing image" });

  try {
    const fd = new URLSearchParams();
    fd.append("image", image);
    if (name) fd.append("name", name);

    const response = await fetch(`https://api.imgbb.com/1/upload?key=${key}`, {
      method: "POST",
      body: fd,
    });
    const data = await response.json();
    res.status(response.status).json(data);
  } catch (e) {
    res.status(500).json({ error: e.message });
  }
}