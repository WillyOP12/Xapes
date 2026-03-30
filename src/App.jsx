import { useState, useEffect, useRef } from "react";
import './index.css';

// ============================================================
// CONSTANTS
// ============================================================
const COLS = 6, ROWS = 8, N = COLS * ROWS;
const GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct";
const PALETTE = ["#e74c3c","#e67e22","#f1c40f","#27ae60","#16a085","#2980b9","#8e44ad","#c0392b","#00bcd4","#ff5722"];

// ============================================================
// STORAGE — Upstash Redis via /api/storage proxy
// ============================================================
function safeParse(v) {
  // Upstash pot retornar valors doble-stringificats si venen del proxy antic
  // Parseja fins que no sigui string o fins que JSON.parse falli
  let result = v;
  let attempts = 0;
  while (typeof result === "string" && attempts < 3) {
    try { result = JSON.parse(result); attempts++; }
    catch { break; }
  }
  return result;
}

const S = {
  async get(k) {
    try {
      const r = await fetch(`/api/storage?op=get&key=${encodeURIComponent(k)}`);
      const d = await r.json();
      if (d.value === null || d.value === undefined) return null;
      return safeParse(d.value);
    } catch(e) { console.error("storage get error:", k, e); return null; }
  },
  async set(k, v) {
    try {
      await fetch("/api/storage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ op: "set", key: k, value: JSON.stringify(v) }),
      });
    } catch(e) { console.error("storage set error:", k, e); }
  },
  async del(k) {
    try {
      await fetch("/api/storage", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ op: "del", key: k }),
      });
    } catch(e) { console.error("storage del error:", k, e); }
  },
};

// ============================================================
// UTILS
// ============================================================
const uid = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 5);
const newSheet = () => ({ id: uid(), cells: Array(N).fill(null) });

async function uploadImgbb(b64, name = "xapa") {
  const r = await fetch("/api/imgbb", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ image: b64, name }),
  });
  const d = await r.json();
  if (!r.ok || !d.success) {
    const msg = d.error?.message || d.error || JSON.stringify(d);
    throw new Error(`ImgBB (${r.status}): ${msg}`);
  }
  return d.data.display_url;
}

async function groqVision(messages, maxTokens = 200) {
  const r = await fetch("/api/groq", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: GROQ_MODEL, max_tokens: maxTokens, messages }),
  });
  const d = await r.json();
  if (d.error) throw new Error(d.error.message || d.error);
  return d.choices?.[0]?.message?.content || "";
}

async function describePin(imageUrl) {
  return groqVision([{
    role: "user",
    content: [
      { type: "image_url", image_url: { url: imageUrl } },
      { type: "text", text: "Descriu aquesta xapa/pin en 2 frases en català. Menciona la forma, els colors principals, qualsevol text o símbol, i l'estil." },
    ],
  }], 150);
}

// ============================================================
// FINGERPRINT — color histogram + average hash, computed locally
// ============================================================
async function computeFingerprint(b64) {
  return new Promise(resolve => {
    try {
      const img = new Image();
      img.onload = () => {
        try {
          // --- Color histogram 8³ = 512 buckets ---
          const S = 32;
          const cv = document.createElement("canvas"); cv.width = cv.height = S;
          const ctx = cv.getContext("2d", { willReadFrequently: true });
          ctx.drawImage(img, 0, 0, S, S);
          const px = ctx.getImageData(0, 0, S, S).data;
          const BINS = 8, hist = new Float32Array(BINS ** 3);
          for (let i = 0; i < px.length; i += 4) {
            const ri = Math.min(7, px[i]   >> 5);
            const gi = Math.min(7, px[i+1] >> 5);
            const bi = Math.min(7, px[i+2] >> 5);
            hist[ri * 64 + gi * 8 + bi]++;
          }
          const total = S * S;
          const histN = Array.from(hist).map(v => v / total);

          // --- Average hash 8x8 ---
          const HS = 8;
          const cv2 = document.createElement("canvas"); cv2.width = cv2.height = HS;
          const ctx2 = cv2.getContext("2d", { willReadFrequently: true });
          ctx2.drawImage(img, 0, 0, HS, HS);
          const px2 = ctx2.getImageData(0, 0, HS, HS).data;
          const grays = [];
          for (let i = 0; i < px2.length; i += 4)
            grays.push(0.299 * px2[i] + 0.587 * px2[i+1] + 0.114 * px2[i+2]);
          const avg = grays.reduce((a, v) => a + v, 0) / grays.length;
          const ahash = grays.map(g => g >= avg ? 1 : 0);

          resolve({ hist: histN, ahash });
        } catch(e) {
          console.error("computeFingerprint error:", e);
          resolve(null);
        }
      };
      img.onerror = (e) => { console.error("img load error", e); resolve(null); };
      img.src = `data:image/jpeg;base64,${b64}`;
    } catch(e) { console.error("computeFingerprint outer error:", e); resolve(null); }
  });
}

function fpSimilarity(a, b) {
  if (!a || !b || !a.hist || !b.hist) return 0;
  let hist = 0;
  for (let i = 0; i < a.hist.length; i++) hist += Math.min(a.hist[i], b.hist[i]);
  let same = 0;
  for (let i = 0; i < a.ahash.length; i++) if (a.ahash[i] === b.ahash[i]) same++;
  const ahash = same / a.ahash.length;
  return 0.65 * hist + 0.35 * ahash;
}

// Phase 1: instant algorithm — falls back to alphabetical if no fingerprints
function algoFindSimilar(queryFp, xapes, topN = 12) {
  if (!queryFp) return [];
  const withFp  = xapes.filter(x => x.fingerprint);
  const withoutFp = xapes.filter(x => !x.fingerprint);
  const scored = withFp
    .map(x => ({ ...x, score: fpSimilarity(queryFp, x.fingerprint) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topN);
  // Append unfingerprinted ones at the end without score
  return [...scored, ...withoutFp.slice(0, Math.max(0, topN - scored.length))];
}

// Phase 2: Groq refines the algo top results
async function aiFindSimilar(queryB64, candidates) {
  const pool = candidates.filter(x => x.description);
  if (!pool.length) return candidates;
  const list = pool.map((x, i) => `[${i}] "${x.name}" (score algo ${x.score?.toFixed(2) ?? "?"}): ${x.description}`).join("\n");
  const text = await groqVision([{
    role: "user",
    content: [
      { type: "image_url", image_url: { url: `data:image/jpeg;base64,${queryB64}` } },
      { type: "text", text: `Reordena aquestes xapes preseleccionades per similitud visual amb la imatge. Retorna ÚNICAMENT un array JSON d'índexs ordenats. Exemple: [2,0,4]\n\n${list}` },
    ],
  }], 80);
  try {
    const idxs = JSON.parse(text.replace(/```\w*|```/g, "").trim());
    const reordered = idxs.map(i => pool[i]).filter(Boolean);
    pool.forEach((x, i) => { if (!idxs.includes(i)) reordered.push(x); });
    return reordered;
  } catch { return candidates; }
}

// ============================================================
// THEME
// ============================================================
const T = {
  bg: "#0b0b10", card: "#13131b", card2: "#181822", border: "#1f1f2e",
  accent: "#f7b731", accentBg: "rgba(247,183,49,0.08)",
  text: "#e2dfd4", muted: "#555570", empty: "#0e0e17",
  success: "#2ecc71", danger: "#e55",
};

const GF = `@import url('https://fonts.googleapis.com/css2?family=Fraunces:wght@700&family=Outfit:wght@300;400;500&display=swap');`;

const INP = { width:"100%", padding:"11px 14px", background:T.empty, border:`1px solid ${T.border}`, borderRadius:8, color:T.text, fontSize:14, outline:"none" };
const LBL = { display:"block", fontSize:11, color:T.muted, marginBottom:6, marginTop:18, textTransform:"uppercase", letterSpacing:"0.07em" };

function btn(v="pri", extra={}) {
  const b = { padding:"10px 20px", borderRadius:8, border:"none", cursor:"pointer", fontWeight:500, fontSize:14, transition:"all .15s", lineHeight:1.4 };
  if (v==="pri")   return { ...b, background:T.accent, color:"#0b0b10", ...extra };
  if (v==="ghost") return { ...b, background:"none", border:`1px solid ${T.border}`, color:T.text, ...extra };
  if (v==="red")   return { ...b, background:"none", border:`1px solid ${T.danger}`, color:T.danger, ...extra };
  return { ...b, background:T.border, color:T.text, ...extra };
}

// ============================================================
// CROPPER — CSS overlay, handles a cantonades i costats, 1:1
// ============================================================
function Cropper({ file, onDone, onCancel }) {
  const imgRef    = useRef();
  const [imgSrc, setImgSrc]   = useState(null);
  const [natW, setNatW]       = useState(0);
  const [natH, setNatH]       = useState(0);
  // crop en píxels de la imatge nativa
  const [crop, setCrop]       = useState(null);
  const drag = useRef(null); // { type, startX, startY, startCrop }

  useEffect(() => {
    const u = URL.createObjectURL(file);
    setImgSrc(u);
    const img = new Image();
    img.onload = () => {
      setNatW(img.naturalWidth);
      setNatH(img.naturalHeight);
      const s = Math.min(img.naturalWidth, img.naturalHeight);
      setCrop({ x: (img.naturalWidth - s) / 2, y: (img.naturalHeight - s) / 2, s });
    };
    img.src = u;
    return () => URL.revokeObjectURL(u);
  }, [file]);

  // Converteix coordenades de pantalla → píxels de la imatge
  const toNat = (clientX, clientY) => {
    const el = imgRef.current;
    if (!el) return { x: 0, y: 0 };
    const r = el.getBoundingClientRect();
    const scX = natW / r.width, scY = natH / r.height;
    return { x: (clientX - r.left) * scX, y: (clientY - r.top) * scY };
  };

  const clamp = (v, min, max) => Math.max(min, Math.min(max, v));

  const onPointerDown = (e, type) => {
    e.preventDefault(); e.stopPropagation();
    drag.current = {
      type,
      startX: e.clientX, startY: e.clientY,
      startCrop: { ...crop },
    };
    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup",   onPointerUp);
  };

  const onPointerMove = (e) => {
    if (!drag.current) return;
    const { type, startX, startY, startCrop: sc } = drag.current;
    const dx = (e.clientX - startX) * (natW / (imgRef.current?.getBoundingClientRect().width || 1));
    const dy = (e.clientY - startY) * (natH / (imgRef.current?.getBoundingClientRect().height || 1));

    setCrop(prev => {
      let { x, y, s } = sc;

      if (type === "move") {
        return {
          s,
          x: clamp(x + dx, 0, natW - s),
          y: clamp(y + dy, 0, natH - s),
        };
      }

      // Handles de cantonada i costat — mantenen 1:1
      // L'ancora és la cantonada/costat oposat
      let newS = s, newX = x, newY = y;

      if (type === "tl") {
        // Àncora: bottom-right
        const ax = x + s, ay = y + s;
        const delta = Math.max(-dx, -dy); // negatiu = encongir
        newS = clamp(s + delta, 20, Math.min(ax, ay));
        newX = ax - newS; newY = ay - newS;
      } else if (type === "tr") {
        // Àncora: bottom-left
        const ay = y + s;
        const delta = Math.max(dx, -dy);
        newS = clamp(s + delta, 20, Math.min(natW - x, ay));
        newX = x; newY = ay - newS;
      } else if (type === "bl") {
        // Àncora: top-right
        const ax = x + s;
        const delta = Math.max(-dx, dy);
        newS = clamp(s + delta, 20, Math.min(ax, natH - y));
        newX = ax - newS; newY = y;
      } else if (type === "br") {
        // Àncora: top-left
        const delta = Math.max(dx, dy);
        newS = clamp(s + delta, 20, Math.min(natW - x, natH - y));
        newX = x; newY = y;
      } else if (type === "t") {
        const ay = y + s;
        newS = clamp(s - dy, 20, ay);
        newX = x + (s - newS) / 2; newY = ay - newS;
      } else if (type === "b") {
        newS = clamp(s + dy, 20, Math.min(natW - x, natH - y));
        newX = x + (s - newS) / 2; newY = y;
      } else if (type === "l") {
        const ax = x + s;
        newS = clamp(s - dx, 20, ax);
        newX = ax - newS; newY = y + (s - newS) / 2;
      } else if (type === "r") {
        newS = clamp(s + dx, 20, Math.min(natW - x, natH - y));
        newX = x; newY = y + (s - newS) / 2;
      }

      // Clamp final dins la imatge
      newX = clamp(newX, 0, natW - newS);
      newY = clamp(newY, 0, natH - newS);
      return { x: newX, y: newY, s: newS };
    });
  };

  const onPointerUp = () => {
    drag.current = null;
    window.removeEventListener("pointermove", onPointerMove);
    window.removeEventListener("pointerup",   onPointerUp);
  };

  const confirm = () => {
    const oc = document.createElement("canvas"); oc.width = oc.height = 400;
    const img = new Image(); img.src = imgSrc;
    img.onload = () => {
      oc.getContext("2d").drawImage(img, crop.x, crop.y, crop.s, crop.s, 0, 0, 400, 400);
      onDone(oc.toDataURL("image/jpeg", 0.88).split(",")[1]);
    };
  };

  if (!imgSrc || !crop || !natW) return (
    <div style={{color:T.muted,textAlign:"center",padding:24}}>Carregant…</div>
  );

  // Percentatges per al CSS
  const pct = v => `${(v / natW * 100).toFixed(4)}%`;
  const pctH = v => `${(v / natH * 100).toFixed(4)}%`;
  const left   = pct(crop.x), top    = pctH(crop.y);
  const width  = pct(crop.s), height = pctH(crop.s);
  const right  = pct(natW - crop.x - crop.s);
  const bottom = pctH(natH - crop.y - crop.s);

  const HA = T.accent; // handle color
  const HOVL = "rgba(0,0,0,0.55)";

  const cornerH = (type, pos) => (
    <div onPointerDown={e => onPointerDown(e, type)}
      style={{ position:"absolute", width:14, height:14, background:HA, borderRadius:3,
        cursor: type==="tl"?"nw-resize":type==="tr"?"ne-resize":type==="bl"?"sw-resize":"se-resize",
        zIndex:3, touchAction:"none", ...pos }} />
  );
  const edgeH = (type, pos, cur) => (
    <div onPointerDown={e => onPointerDown(e, type)}
      style={{ position:"absolute", background:HA, borderRadius:2, zIndex:3, touchAction:"none", cursor:cur, ...pos }} />
  );

  return (
    <div>
      <p style={{color:T.muted,fontSize:12,marginBottom:10,textAlign:"center"}}>
        Arrossega les cantonades o costats per retallar · centre per moure
      </p>

      {/* CONTENIDOR */}
      <div style={{position:"relative",userSelect:"none",borderRadius:8,overflow:"hidden",touchAction:"none"}}>
        <img ref={imgRef} src={imgSrc} alt="" draggable={false}
          style={{width:"100%",height:"auto",display:"block"}} />

        {/* OVERLAY FOSC: 4 tires */}
        <div style={{position:"absolute",inset:0,pointerEvents:"none"}}>
          {/* top */}
          <div style={{position:"absolute",top:0,left:0,right:0,height:top,background:HOVL}}/>
          {/* bottom */}
          <div style={{position:"absolute",bottom:0,left:0,right:0,height:bottom,background:HOVL}}/>
          {/* left */}
          <div style={{position:"absolute",top,bottom,left:0,width:left,background:HOVL}}/>
          {/* right */}
          <div style={{position:"absolute",top,bottom,right:0,width:right,background:HOVL}}/>
        </div>

        {/* BOX DE RETALL */}
        <div onPointerDown={e => onPointerDown(e, "move")}
          style={{position:"absolute",left,top,width,height,
            border:`2px solid ${HA}`,cursor:"move",boxSizing:"border-box",touchAction:"none",zIndex:2}}>

          {/* Regla dels terços */}
          <div style={{position:"absolute",inset:0,pointerEvents:"none"}}>
            {[1,2].map(i=><>
              <div key={`v${i}`} style={{position:"absolute",left:`${i*33.33}%`,top:0,bottom:0,width:1,background:"rgba(255,255,255,0.2)"}}/>
              <div key={`h${i}`} style={{position:"absolute",top:`${i*33.33}%`,left:0,right:0,height:1,background:"rgba(255,255,255,0.2)"}}/>
            </>)}
          </div>

          {/* Cantonades */}
          {cornerH("tl", {top:-7,left:-7})}
          {cornerH("tr", {top:-7,right:-7})}
          {cornerH("bl", {bottom:-7,left:-7})}
          {cornerH("br", {bottom:-7,right:-7})}

          {/* Costats (barra fina al mig) */}
          {edgeH("t",  {top:-4,left:"50%",transform:"translateX(-50%)",width:28,height:8},  "n-resize")}
          {edgeH("b",  {bottom:-4,left:"50%",transform:"translateX(-50%)",width:28,height:8}, "s-resize")}
          {edgeH("l",  {left:-4,top:"50%",transform:"translateY(-50%)",width:8,height:28},   "w-resize")}
          {edgeH("r",  {right:-4,top:"50%",transform:"translateY(-50%)",width:8,height:28},  "e-resize")}
        </div>
      </div>

      <div style={{display:"flex",gap:8,marginTop:14}}>
        <button onClick={onCancel} style={btn("def",{flex:1})}>Cancel·lar</button>
        <button onClick={confirm}  style={btn("pri",{flex:1})}>Retallar ✓</button>
      </div>
    </div>
  );
}

// Module-level active stream — guaranteed kill from anywhere
let _activeStream = null;
function stopAllCameras() {
  if (_activeStream) { _activeStream.getTracks().forEach(t => t.stop()); _activeStream = null; }
}

// ============================================================
// CAMERA
// ============================================================
function CameraCapture({ onCapture, onCancel }) {
  const videoRef    = useRef();
  const mountedRef  = useRef(true);
  const [devices, setDevices]       = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [err, setErr]               = useState("");
  const [ready, setReady]           = useState(false);

  const startCamera = async (deviceId) => {
    stopAllCameras(); if (!mountedRef.current) return;
    setReady(false); setErr("");
    try {
      // Si és capturadora HDMI (o qualsevol dispositiu sense facing mode),
      // no posem facingMode ni resolució — deixem que el dispositiu mani
      const video = deviceId
        ? { deviceId: { exact: deviceId } }
        : true; // true = cap constraint, agafa el primer dispositiu disponible
      const stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
      if (!mountedRef.current) { stream.getTracks().forEach(t => t.stop()); return; }
      _activeStream = stream;
      const v = videoRef.current; if (!v) return;
      v.srcObject = stream;
      await new Promise((res, rej) => { v.onloadedmetadata = res; v.onerror = rej; });
      if (!mountedRef.current) return;
      await v.play();
      if (!mountedRef.current) return;
      setReady(true);
      const devs = await navigator.mediaDevices.enumerateDevices();
      const cams = devs.filter(d => d.kind === "videoinput");
      if (!mountedRef.current) return;
      setDevices(cams);
      if (!deviceId && cams.length) setSelectedId(cams[0].deviceId);
    } catch(e) {
      if (!mountedRef.current) return;
      if (e.name === "AbortError") return;
      setErr(e.name === "NotAllowedError"
        ? "Permís denegat. Permet la càmera des de la configuració del navegador."
        : `Error: ${e.message}`);
    }
  };

  useEffect(() => {
    mountedRef.current = true;
    startCamera("");
    return () => { mountedRef.current = false; stopAllCameras(); };
  }, []);

  const changeDevice = async id => { setSelectedId(id); await startCamera(id); };

  const capture = () => {
    const v = videoRef.current; if (!v) return;
    const cv = document.createElement("canvas");
    cv.width = v.videoWidth; cv.height = v.videoHeight;
    cv.getContext("2d").drawImage(v, 0, 0);
    const b64 = cv.toDataURL("image/jpeg", 0.9).split(",")[1];
    stopAllCameras();
    const bytes = Uint8Array.from(atob(b64), c => c.charCodeAt(0));
    const blob = new Blob([bytes], { type: "image/jpeg" });
    onCapture(new File([blob], "camera.jpg", { type: "image/jpeg" }));
  };

  return (
    <div>
      {err
        ? <div style={{textAlign:"center",padding:"24px 0"}}>
            <div style={{fontSize:36,marginBottom:10}}>🚫</div>
            <p style={{color:T.danger,fontSize:14,marginBottom:20,lineHeight:1.6}}>{err}</p>
            <button onClick={onCancel} style={btn("def",{width:"100%"})}>Tornar</button>
          </div>
        : <>
          {devices.length > 1 && (
            <div style={{marginBottom:10}}>
              <label style={LBL}>Càmera</label>
              <select value={selectedId} onChange={e => changeDevice(e.target.value)}
                style={{...INP, cursor:"pointer"}}>
                {devices.map((d, i) => <option key={d.deviceId} value={d.deviceId}>{d.label || `Càmera ${i+1}`}</option>)}
              </select>
            </div>
          )}
          <div style={{position:"relative",background:"#000",borderRadius:12,overflow:"hidden",marginBottom:12,display:"flex",alignItems:"center",justifyContent:"center",minHeight:180}}>
            <video ref={videoRef} playsInline muted
              style={{width:"100%",height:"auto",display:"block",maxHeight:"55vh"}}/>
            {!ready && <div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",color:T.muted,fontSize:13}}>Iniciant càmera…</div>}
          </div>
          <div style={{display:"flex",gap:8}}>
            <button onClick={onCancel} style={btn("def",{flex:1})}>Cancel·lar</button>
            <button onClick={capture} disabled={!ready}
              style={btn("pri",{flex:2,opacity:ready?1:0.4,cursor:ready?"pointer":"not-allowed"})}>
              📸 Capturar
            </button>
          </div>
        </>
      }
    </div>
  );
}

// ============================================================
// MODAL
// ============================================================
function Modal({ title, onClose, children, maxW=440 }) {
  const isJSX = title && typeof title === "object";
  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.88)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:999,padding:16}}
      onClick={e=>e.target===e.currentTarget&&onClose()}>
      <div style={{background:T.card,borderRadius:16,padding:24,width:"100%",maxWidth:maxW,maxHeight:"92vh",overflowY:"auto",border:`1px solid ${T.border}`}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:20}}>
          {isJSX
            ? <>{title}<button onClick={onClose} style={{background:"none",border:"none",color:T.muted,fontSize:22,cursor:"pointer",lineHeight:1,padding:"0 2px",marginLeft:8}}>×</button></>
            : <><h3 style={{fontFamily:"Fraunces",fontSize:20,color:T.text,flex:1}}>{title}</h3>
                <button onClick={onClose} style={{background:"none",border:"none",color:T.muted,fontSize:22,cursor:"pointer",lineHeight:1,padding:"0 2px"}}>×</button></>
          }
        </div>
        {children}
      </div>
    </div>
  );
}

// ============================================================
// APP ROOT
// ============================================================

// --- URL helpers ---
function getParams() { return new URLSearchParams(window.location.search); }
function pushUrl(params) {
  const s = params.toString();
  window.history.pushState({}, "", s ? `?${s}` : window.location.pathname);
}
function replaceUrl(params) {
  const s = params.toString();
  window.history.replaceState({}, "", s ? `?${s}` : window.location.pathname);
}

export default function App() {
  const [albums, setAlbums]       = useState([]);
  const [loading, setLoading]     = useState(true);
  const [view, setView]           = useState("home");
  const [album, setAlbum]         = useState(null);
  const [sheetIdx, setSheetIdx]   = useState(0);
  const [data, setData]           = useState(null);
  const [modal, setModal]         = useState(null);
  const [mv, setMv]               = useState(null);
  const [busy, setBusy]           = useState("");
  const [searchOpen, setSearchOpen] = useState(false);

  // --- Carrega inicial + restaura URL ---
  useEffect(() => {
    (async () => {
      const a = await S.get("xapes-albums");
      const loadedAlbums = Array.isArray(a) ? a : [];
      setAlbums(loadedAlbums);
      setLoading(false);
      await restoreFromUrl(loadedAlbums);
    })();
  }, []);

  const restoreFromUrl = async (loadedAlbums) => {
    const p = getParams();
    const albumId  = p.get("album");
    const fulla    = p.get("fulla");
    const page     = p.get("page");
    const cercaQ   = p.get("cerca");
    const tagQ     = p.get("tag");
    const xapaId   = p.get("xapa");
    const imageB64 = p.get("image");
    const nouFlag  = p.get("nou");

    if (!albumId) return;
    const found = (loadedAlbums || albums).find(a => a.id === albumId);
    if (!found) return;
    const raw = await S.get(`xapes-ad-${found.id}`);
    const d = (raw && typeof raw === "object" && !Array.isArray(raw))
      ? raw : { sheets: [], bigItems: [] };
    if (!Array.isArray(d.sheets))   d.sheets   = [];
    if (!Array.isArray(d.bigItems)) d.bigItems = [];
    setAlbum(found); setData(d);

    // Navega a la fulla
    if (fulla !== null) {
      const idx = Math.max(0, parseInt(fulla, 10) - 1) || 0;
      setSheetIdx(idx); setView("sheet");
    } else {
      setView("album");
    }

    // ?xapa=id — obre la xapa directament
    if (xapaId) {
      let fcell = null, fidx = null, fsi = null, fbig = null, fbi = null;
      for (let si = 0; si < d.sheets.length; si++) {
        const ci = d.sheets[si].cells.findIndex(c => c?.id === xapaId);
        if (ci !== -1) { fcell = d.sheets[si].cells[ci]; fidx = ci; fsi = si; break; }
      }
      if (!fcell) {
        const bi = (d.bigItems || []).findIndex(c => c?.id === xapaId);
        if (bi !== -1) { fbig = d.bigItems[bi]; fbi = bi; }
      }
      if (fcell) {
        setSheetIdx(fsi); setView("sheet");
        setTimeout(() => setModal({ t:"view", cell: fcell, idx: fidx }), 120);
      } else if (fbig) {
        setView("album");
        setTimeout(() => setModal({ t:"viewBig", item: fbig, idx: fbi }), 120);
      }
    }

    // ?image=b64 — obre cerca visual amb imatge precarregada
    if (imageB64) {
      setInitImage(decodeURIComponent(imageB64));
      setSearchOpen(true);
    }

    // ?cerca= o ?tag= — obre cerca text
    if ((page === "cerca" || cercaQ || tagQ) && !imageB64) {
      setSearchOpen(true);
      if (cercaQ) setInitSearch(cercaQ);
      if (tagQ)   setInitSearch(tagQ);
    }

    // ?nou=1 + ?foto= + ?nom= — obre modal d'afegir preomplert
    const fotoParam  = p.get("foto");
    const nomParam   = p.get("nom");
    const tagsParam  = p.get("tags");
    const tipusParam = p.get("tipus");
    const estatParam = p.get("estat");
    const anyParam   = p.get("any");
    const origenParam= p.get("origen");
    const fabParam   = p.get("fabricant");
    const matParam   = p.get("material");
    const midaParam  = p.get("mida");
    const limitParam = p.get("limitada");
    const numerParam = p.get("numeracio");
    const adqParam   = p.get("adquirit");
    const dataParam  = p.get("data");
    const preuParam  = p.get("preu");
    const valorParam = p.get("valor");
    if (nouFlag === "1") {
      const modalData = { t:"add", idx: 0 };
      if (fotoParam)  modalData.initFoto      = decodeURIComponent(fotoParam);
      if (nomParam)   modalData.initNom       = decodeURIComponent(nomParam);
      if (tagsParam)  modalData.initTags      = decodeURIComponent(tagsParam);
      if (tipusParam) modalData.initTipus     = decodeURIComponent(tipusParam);
      if (estatParam) modalData.initEstat     = decodeURIComponent(estatParam);
      if (anyParam)   modalData.initAny       = decodeURIComponent(anyParam);
      if (origenParam)modalData.initOrigen    = decodeURIComponent(origenParam);
      if (fabParam)   modalData.initFabricant = decodeURIComponent(fabParam);
      if (matParam)   modalData.initMaterial  = decodeURIComponent(matParam);
      if (midaParam)  modalData.initMida      = decodeURIComponent(midaParam);
      if (limitParam) modalData.initLimitada  = limitParam === "1";
      if (numerParam) modalData.initNumeracio = decodeURIComponent(numerParam);
      if (adqParam)   modalData.initAdquirit  = decodeURIComponent(adqParam);
      if (dataParam)  modalData.initDataAdq   = decodeURIComponent(dataParam);
      if (preuParam)  modalData.initPreuPagat = decodeURIComponent(preuParam);
      if (valorParam) modalData.initValorEst  = decodeURIComponent(valorParam);
      setTimeout(() => setModal(modalData), 150);
    }
  };

  const [initSearch, setInitSearch] = useState("");
  const [initImage,  setInitImage]  = useState(null);

  // --- Escolta el botó enrere del navegador ---
  useEffect(() => {
    const onPop = () => restoreFromUrl(albums);
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, [albums]);

  // --- Sincronitza URL quan canvia l'estat ---
  useEffect(() => {
    if (loading) return;
    const p = new URLSearchParams();
    if (album)              p.set("album", album.id);
    if (view === "sheet")   p.set("fulla", String(sheetIdx + 1));
    if (searchOpen)         p.set("page",  "cerca");
    if (modal?.t === "view"    && modal.cell?.id) p.set("xapa", modal.cell.id);
    if (modal?.t === "viewBig" && modal.item?.id) p.set("xapa", modal.item.id);
    if (modal?.t === "add")     p.set("nou", "1");
    replaceUrl(p);
  }, [view, album, sheetIdx, searchOpen, modal, loading]);

  const saveAlbums = async a => { setAlbums(a); await S.set("xapes-albums", a); };
  const saveData   = async (aid, d) => { setData(d); await S.set(`xapes-ad-${aid}`, d); };

  const openAlbum = async a => {
    const raw = await S.get(`xapes-ad-${a.id}`);
    const d = (raw && typeof raw === "object" && !Array.isArray(raw))
      ? raw : { sheets: [], bigItems: [] };
    if (!Array.isArray(d.sheets))   d.sheets   = [];
    if (!Array.isArray(d.bigItems)) d.bigItems = [];
    setAlbum(a); setData(d); setView("album");
  };

  const goBack = () => {
    if (view === "sheet") { setView("album"); setMv(null); }
    else { setView("home"); setAlbum(null); setData(null); }
  };

  const curSheet  = () => data?.sheets?.[sheetIdx];
  const openSheet = i  => { setSheetIdx(i); setView("sheet"); };
  const goPrev    = () => setSheetIdx(i => Math.max(0, i - 1));
  const goNext    = () => setSheetIdx(i => Math.min((data?.sheets?.length || 1) - 1, i + 1));

  const handleCellClick = async idx => {
    const cs = curSheet(); if (!cs) return;
    const cell = cs.cells[idx];

    if (mv) {
      let newSheets = data.sheets;
      let newBigItems = [...(data.bigItems||[])];
      if (mv.type==="cell") {
        newSheets = data.sheets.map((s,si)=>{
          if (si!==sheetIdx) return s;
          const c=[...s.cells]; [c[mv.idx],c[idx]]=[c[idx],c[mv.idx]]; return {...s,cells:c};
        });
      } else {
        let displaced=null;
        newSheets = data.sheets.map((s,si)=>{
          if (si!==sheetIdx) return s;
          const c=[...s.cells]; displaced=c[idx]; c[idx]=mv.item; return {...s,cells:c};
        });
        if (displaced) newBigItems[mv.bigIdx]=displaced;
        else newBigItems=newBigItems.filter((_,i)=>i!==mv.bigIdx);
      }
      const nd={...data,sheets:newSheets,bigItems:newBigItems};
      saveData(album.id, nd); setMv(null);
    } else if (cell) {
      setModal({t:"view",cell,idx});
    } else {
      setModal({t:"add",idx});
    }
  };

  const allXapes = () => {
    const out = [];
    const d = data || {};
    d.sheets?.forEach(s => s.cells?.forEach((c,i) => c && out.push({...c, albumName:album?.name, albumId:album?.id, type:"cell"})));
    d.bigItems?.forEach((c,i) => c && out.push({...c, albumName:album?.name, albumId:album?.id, bigIdx:i, type:"big"}));
    return out;
  };

  if (loading) return (
    <div style={{display:"flex",flexDirection:"column",alignItems:"center",justifyContent:"center",minHeight:"100vh",gap:16}}>
      <style>{GF}</style>
      <img src="/pin.svg" alt="Xapes" style={{width:72,height:72}}/>
      <p style={{fontFamily:"Fraunces",fontSize:24,color:T.accent}}>Xapes</p>
      <p style={{color:T.muted,fontSize:13}}>Carregant col·lecció…</p>
    </div>
  );

  return (
    <div style={{minHeight:"100vh"}}>
      <style>{GF}</style>
      {/* NAVBAR */}
      <div style={{background:T.card,borderBottom:`1px solid ${T.border}`,display:"flex",alignItems:"center",gap:8,padding:"0 16px",height:52,position:"sticky",top:0,zIndex:100}}>
        {view!=="home"&&(
          <button onClick={goBack} style={{background:"none",border:"none",color:T.muted,cursor:"pointer",fontSize:20,padding:"4px 6px",lineHeight:1}}>←</button>
        )}
        <span style={{fontFamily:"Fraunces",fontSize:21,color:T.accent,flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>
          {view==="home"
            ? <><img src="/pin.svg" alt="" style={{width:20,height:20,marginRight:6,verticalAlign:"middle"}}/>Xapes</>
            : view==="album" ? album?.name
            : `${album?.name} · Fulla ${sheetIdx+1}`}
        </span>
        {(view==="album"||view==="sheet")&&(
          <button onClick={()=>setSearchOpen(true)} title="Cerca" style={{background:"none",border:"none",color:T.muted,cursor:"pointer",fontSize:17,padding:"4px 6px"}}>🔍</button>
        )}
        <button onClick={()=>setModal({t:"editAlbum",album})} title="Editar" style={{display:view==="album"?"block":"none",background:"none",border:"none",color:T.muted,cursor:"pointer",fontSize:15,padding:"4px 6px"}}>✎</button>
      </div>

      <div style={{padding:"20px 16px",maxWidth:860,margin:"0 auto"}}>
        {view==="home"&&(
          <HomeV albums={albums} onOpen={openAlbum}
            onAdd={()=>setModal({t:"addAlbum"})}
            onEdit={a=>setModal({t:"editAlbum",album:a})}/>
        )}
        {view==="album"&&data&&(
          <AlbumV album={album} data={data}
            onOpenSheet={(_,i)=>openSheet(i)}
            onAddSheet={()=>{
              const s=newSheet();
              const nd={...data,sheets:[...data.sheets,s]};
              saveData(album.id,nd);
              openSheet(nd.sheets.length-1);
            }}
            onAddBig={()=>setModal({t:"addBig"})}
            onViewBig={(item,i)=>setModal({t:"viewBig",item,idx:i})}/>
        )}
        {view==="sheet"&&data&&curSheet()&&(
          <SheetV sheet={curSheet()} sheetIdx={sheetIdx} totalSheets={data.sheets.length}
            mv={mv} onSetMv={setMv} onCell={handleCellClick}
            onPrev={goPrev} onNext={goNext}/>
        )}
      </div>

      {/* MODALS */}
      {modal?.t==="addAlbum"&&(
        <AlbumModal onClose={()=>setModal(null)} onSave={({name,color})=>{
          const a={id:uid(),name,color};
          saveAlbums([...albums,a]);
          S.set(`xapes-ad-${a.id}`,{sheets:[],bigItems:[]});
          setModal(null);
        }}/>
      )}
      {modal?.t==="editAlbum"&&modal.album&&(
        <AlbumModal init={modal.album} onClose={()=>setModal(null)} onSave={({name,color})=>{
          saveAlbums(albums.map(a=>a.id===modal.album.id?{...a,name,color}:a));
          if(album?.id===modal.album.id) setAlbum(p=>({...p,name,color}));
          setModal(null);
        }}/>
      )}
      {modal?.t==="add"&&(
        <AddXapaModal title="Afegir xapa" busy={busy} setBusy={setBusy} onClose={()=>setModal(null)}
          initNom={modal.initNom} initFoto={modal.initFoto}
          initTags={modal.initTags}
          initTipus={modal.initTipus} initEstat={modal.initEstat}
          initAny={modal.initAny} initOrigen={modal.initOrigen}
          initFabricant={modal.initFabricant} initMaterial={modal.initMaterial} initMida={modal.initMida}
          initLimitada={modal.initLimitada} initNumeracio={modal.initNumeracio}
          initAdquirit={modal.initAdquirit} initDataAdq={modal.initDataAdq}
          initPreuPagat={modal.initPreuPagat} initValorEst={modal.initValorEst}
          onSave={async(xapa)=>{
            const nd={...data,sheets:data.sheets.map((s,si)=>si!==sheetIdx?s:(()=>{const c=[...s.cells];c[modal.idx]={id:uid(),...xapa};return{...s,cells:c};})())};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="view"&&(
        <ViewXapaModal cell={modal.cell} idx={modal.idx}
          cells={curSheet()?.cells||[]}
          onClose={()=>setModal(null)}
          onEdit={()=>setModal({t:"edit", cell:modal.cell, idx:modal.idx})}
          onMove={()=>{setMv({type:"cell",idx:modal.idx,item:modal.cell});setModal(null);}}
          onNavigate={idx=>setModal({t:"view", cell:curSheet().cells[idx], idx})}
          onDelete={()=>{
            const nd={...data,sheets:data.sheets.map((s,si)=>si!==sheetIdx?s:(()=>{const c=[...s.cells];c[modal.idx]=null;return{...s,cells:c};})())};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="edit"&&(
        <AddXapaModal title="Editar xapa" busy={busy} setBusy={setBusy}
          initCell={modal.cell} onClose={()=>setModal(null)}
          onSave={async(xapa)=>{
            const nd={...data,sheets:data.sheets.map((s,si)=>si!==sheetIdx?s:(()=>{const c=[...s.cells];c[modal.idx]=xapa;return{...s,cells:c};})())};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="addBig"&&(
        <AddXapaModal title="Afegir xapa gran" busy={busy} setBusy={setBusy} onClose={()=>setModal(null)}
          onSave={async(xapa)=>{
            const nd={...data,bigItems:[...(data.bigItems||[]),{id:uid(),...xapa}]};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="viewBig"&&(
        <ViewXapaModal cell={modal.item} onClose={()=>setModal(null)}
          onEdit={()=>setModal({t:"editBig", item:modal.item, idx:modal.idx})}
          onMove={()=>{
            if(!data.sheets.length){alert("Crea una fulla primer!");return;}
            setMv({type:"big",bigIdx:modal.idx,item:modal.item});
            openSheet(0); setModal(null);
          }}
          onDelete={()=>{
            const nd={...data,bigItems:(data.bigItems||[]).filter((_,i)=>i!==modal.idx)};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="editBig"&&(
        <AddXapaModal title="Editar xapa gran" busy={busy} setBusy={setBusy}
          initCell={modal.item} onClose={()=>setModal(null)}
          onSave={async(xapa)=>{
            const nd={...data,bigItems:(data.bigItems||[]).map((it,i)=>i===modal.idx?xapa:it)};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {searchOpen&&<SearchModal
        onClose={()=>{setSearchOpen(false);setInitSearch("");setInitImage(null);}}
        getAllXapes={allXapes}
        initQuery={initSearch}
        initImage={initImage}
      />}
      <Analytics />
      <SpeedInsights />
    </div>
  );
}

// ============================================================
// HOME VIEW
// ============================================================
function HomeV({ albums, onOpen, onAdd, onEdit }) {
  return (
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:24}}>
        <h2 style={{fontFamily:"Fraunces",fontSize:28}}>Col·leccions</h2>
        <button onClick={onAdd} style={btn("pri")}>+ Nou àlbum</button>
      </div>
      {!albums.length
        ?<div style={{textAlign:"center",padding:"60px 20px",color:T.muted}}>
          <div style={{fontSize:48,marginBottom:12}}>🗂️</div>
          <p>Crea el teu primer àlbum de xapes!</p>
        </div>
        :<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(170px,1fr))",gap:14}}>
          {albums.map(a=>(
            <div key={a.id} onClick={()=>onOpen(a)}
              style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:14,padding:20,cursor:"pointer",transition:"all .2s",position:"relative"}}
              onMouseEnter={e=>{e.currentTarget.style.borderColor=a.color;e.currentTarget.style.transform="translateY(-3px)";}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor=T.border;e.currentTarget.style.transform="none";}}>
              <div style={{width:44,height:44,borderRadius:10,background:a.color,marginBottom:14}}/>
              <h3 style={{fontFamily:"Fraunces",fontSize:17,marginBottom:2,paddingRight:24}}>{a.name}</h3>
              <button onClick={e=>{e.stopPropagation();onEdit(a);}}
                style={{position:"absolute",top:10,right:10,background:"none",border:"none",color:T.muted,cursor:"pointer",fontSize:15,padding:4}}>✎</button>
            </div>
          ))}
        </div>
      }
    </div>
  );
}

// ============================================================
// ALBUM VIEW — carrusel de fulles
// ============================================================
function AlbumV({ album, data, onOpenSheet, onAddSheet, onAddBig, onViewBig }) {
  const { sheets, bigItems=[] } = data;
  return (
    <div>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
        <h2 style={{fontFamily:"Fraunces",fontSize:26}}>Fulles</h2>
        <button onClick={onAddSheet} style={btn("pri")}>+ Nova fulla</button>
      </div>
      {!sheets.length
        ?<div style={{textAlign:"center",padding:40,color:T.muted,background:T.card,borderRadius:12,marginBottom:28}}>Cap fulla. Crea-ne una!</div>
        :<div style={{display:"flex",gap:14,overflowX:"auto",paddingBottom:12,marginBottom:28,scrollSnapType:"x mandatory",WebkitOverflowScrolling:"touch",scrollbarWidth:"thin",scrollbarColor:`${T.border} transparent`}}>
          {sheets.map((s,i)=>(
            <div key={s.id} onClick={()=>onOpenSheet(s,i)}
              style={{flex:"0 0 140px",background:T.card,border:`1px solid ${T.border}`,borderRadius:14,padding:12,cursor:"pointer",transition:"all .2s",scrollSnapAlign:"start",userSelect:"none"}}
              onMouseEnter={e=>{e.currentTarget.style.borderColor=album.color;e.currentTarget.style.transform="translateY(-3px)";}}
              onMouseLeave={e=>{e.currentTarget.style.borderColor=T.border;e.currentTarget.style.transform="none";}}>
              <MiniGrid cells={s.cells} color={album.color}/>
              <p style={{fontSize:13,color:T.text,marginTop:9,fontWeight:500}}>Fulla {i+1}</p>
              <p style={{fontSize:11,color:T.muted}}>{s.cells.filter(Boolean).length}/{N} xapes</p>
            </div>
          ))}
        </div>
      }
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:14}}>
        <h2 style={{fontFamily:"Fraunces",fontSize:22}}>Xapes grans</h2>
        <button onClick={onAddBig} style={btn("def")}>+ Afegir</button>
      </div>
      {!bigItems.length
        ?<div style={{textAlign:"center",padding:28,color:T.muted,background:T.card,borderRadius:12}}>Cap xapa gran</div>
        :<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(110px,1fr))",gap:12}}>
          {bigItems.map((item,i)=>item&&(
            <div key={item.id||i} onClick={()=>onViewBig(item,i)}
              style={{background:T.card,border:`1px solid ${T.border}`,borderRadius:12,overflow:"hidden",cursor:"pointer",transition:"border .15s"}}
              onMouseEnter={e=>e.currentTarget.style.borderColor=album.color}
              onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>
              {item.imageUrl&&<img src={item.imageUrl} alt={item.name} style={{width:"100%",aspectRatio:"1/1",objectFit:"cover",display:"block"}}/>}
              <p style={{padding:"7px 9px",fontSize:12,color:T.text,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{item.name}</p>
            </div>
          ))}
        </div>
      }
    </div>
  );
}

function MiniGrid({ cells, color }) {
  return (
    <div style={{display:"grid",gridTemplateColumns:`repeat(${COLS},1fr)`,gap:1.5,width:"100%",aspectRatio:`${COLS}/${ROWS}`}}>
      {cells.map((c,i)=><div key={i} style={{background:c?color:"#1c1c28",borderRadius:1}}/>)}
    </div>
  );
}

// ============================================================
// SHEET VIEW — slides
// ============================================================
function SheetV({ sheet, sheetIdx, totalSheets, mv, onSetMv, onCell, onPrev, onNext }) {
  return (
    <div>
      {/* SLIDE NAV */}
      <div style={{display:"flex",alignItems:"center",justifyContent:"center",gap:16,marginBottom:16}}>
        <button onClick={onPrev} disabled={sheetIdx===0}
          style={{background:"none",border:`1px solid ${T.border}`,color:sheetIdx===0?T.muted:T.text,borderRadius:8,width:38,height:38,fontSize:18,cursor:sheetIdx===0?"default":"pointer",display:"flex",alignItems:"center",justifyContent:"center",transition:"all .15s"}}
          onMouseEnter={e=>{if(sheetIdx>0)e.currentTarget.style.borderColor=T.accent;}}
          onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>‹</button>
        <div style={{textAlign:"center"}}>
          <span style={{fontFamily:"Fraunces",fontSize:20,color:T.accent}}>Fulla {sheetIdx+1}</span>
          <span style={{color:T.muted,fontSize:13,marginLeft:6}}>/ {totalSheets}</span>
        </div>
        <button onClick={onNext} disabled={sheetIdx===totalSheets-1}
          style={{background:"none",border:`1px solid ${T.border}`,color:sheetIdx===totalSheets-1?T.muted:T.text,borderRadius:8,width:38,height:38,fontSize:18,cursor:sheetIdx===totalSheets-1?"default":"pointer",display:"flex",alignItems:"center",justifyContent:"center",transition:"all .15s"}}
          onMouseEnter={e=>{if(sheetIdx<totalSheets-1)e.currentTarget.style.borderColor=T.accent;}}
          onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>›</button>
      </div>
      {/* DOTS */}
      {totalSheets>1&&(
        <div style={{display:"flex",justifyContent:"center",gap:5,marginBottom:16,flexWrap:"wrap"}}>
          {Array.from({length:totalSheets},(_,i)=>(
            <div key={i} style={{width:i===sheetIdx?18:7,height:7,borderRadius:4,background:i===sheetIdx?T.accent:T.border,transition:"all .25s"}}/>
          ))}
        </div>
      )}
      {/* MOVE BANNER */}
      {mv&&(
        <div style={{background:T.accentBg,border:`1px solid ${T.accent}`,borderRadius:10,padding:"10px 14px",marginBottom:14,display:"flex",justifyContent:"space-between",alignItems:"center"}}>
          <span style={{color:T.accent,fontSize:14}}>
            Mou <b>"{mv.item?.name}"</b> — toca la casella destí
            <span style={{fontSize:11,color:T.muted,marginLeft:10}}>🟢 buit · 🔴 intercanvi</span>
          </span>
          <button onClick={()=>onSetMv(null)} style={{background:"none",border:"none",color:T.accent,cursor:"pointer",fontSize:20,lineHeight:1}}>×</button>
        </div>
      )}
      {/* GRID */}
      <div style={{display:"grid",gridTemplateColumns:`repeat(${COLS},1fr)`,gap:5,background:T.card2,padding:10,borderRadius:16,border:`1px solid ${T.border}`,maxWidth:500,margin:"0 auto"}}>
        {sheet.cells.map((cell,i)=>{
          const isOrigin=mv?.type==="cell"&&mv?.idx===i;
          const isTarget=mv&&!isOrigin;
          return (
            <div key={i} onClick={()=>onCell(i)} title={cell?.name||""}
              style={{aspectRatio:"1",borderRadius:7,overflow:"hidden",cursor:"pointer",
                border:`2px solid ${isOrigin?T.accent:isTarget?(cell?T.danger:T.success):"transparent"}`,
                background:cell?"#000":T.empty,position:"relative",transition:"border .12s"}}>
              {cell?.imageUrl&&<img src={cell.imageUrl} alt={cell.name} style={{width:"100%",height:"100%",objectFit:"cover",display:"block"}}/>}
              {!cell&&!mv&&<div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",color:"#22222e",fontSize:16}}>+</div>}
              {mv&&isTarget&&<div style={{position:"absolute",inset:0,display:"flex",alignItems:"center",justifyContent:"center",background:"rgba(0,0,0,0.55)",fontSize:20}}>{cell?"⇄":"↓"}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}

// ============================================================
// ALBUM MODAL
// ============================================================
function AlbumModal({ init, onClose, onSave }) {
  const [name, setName] = useState(init?.name||"");
  const [color, setColor] = useState(init?.color||PALETTE[0]);
  return (
    <Modal title={init?"Editar àlbum":"Nou àlbum"} onClose={onClose}>
      <label style={LBL}>Nom</label>
      <input value={name} onChange={e=>setName(e.target.value)} placeholder="Ex: Animals, CUP, Viatge..." style={INP} autoFocus
        onKeyDown={e=>e.key==="Enter"&&name.trim()&&onSave({name:name.trim(),color})}/>
      <label style={LBL}>Color</label>
      <div style={{display:"flex",gap:8,flexWrap:"wrap",margin:"8px 0 22px"}}>
        {PALETTE.map(c=>(
          <div key={c} onClick={()=>setColor(c)}
            style={{width:34,height:34,borderRadius:8,background:c,cursor:"pointer",border:`3px solid ${color===c?"#fff":"transparent"}`,transition:"border .12s"}}/>
        ))}
      </div>
      <div style={{display:"flex",gap:8}}>
        <button onClick={onClose} style={btn("def",{flex:1})}>Cancel·lar</button>
        <button onClick={()=>name.trim()&&onSave({name:name.trim(),color})} style={btn("pri",{flex:1})}>Desar</button>
      </div>
    </Modal>
  );
}


// ============================================================
// ADD XAPA MODAL
// ============================================================
function AddXapaModal({ title, busy, setBusy, onClose, onSave,
  initNom="", initFoto=null, initTags="", initCell=null
}) {
  const isEdit = !!initCell;
  const [name,     setName]    = useState(isEdit ? initCell.name                  : initNom);
  const [tags,     setTags]    = useState(isEdit ? (initCell.tags||[]).join(", ") : initTags);
  const [step,     setStep]    = useState("form");
  const [file,     setFile]    = useState(null);
  const [b64,      setB64]     = useState(null);
  const [prev,     setPrev]    = useState(isEdit ? initCell.imageUrl : null);
  const [urlInput, setUrlInput]= useState("");
  const [urlMode,  setUrlMode] = useState(false);
  const [dragOver, setDragOver]= useState(false);
  const fileRef = useRef(), dropRef = useRef();

  useEffect(() => {
    if (!initFoto || isEdit) return;
    (async () => {
      setBusy("Carregant foto…");
      try {
        if (!initFoto.startsWith("http")) {
          const d = initFoto.includes(",") ? initFoto.split(",")[1] : initFoto;
          setB64(d); setPrev(`data:image/jpeg;base64,${d}`);
        } else {
          const r = await fetch(initFoto);
          const blob = await r.blob();
          setFile(new File([blob], "foto.jpg", { type: blob.type })); setStep("crop");
        }
      } catch(e) { alert("Error carregant foto: " + e.message); }
      finally { setBusy(""); }
    })();
  }, []);

  const onCrop   = d => { setB64(d); setPrev(`data:image/jpeg;base64,${d}`); setStep("form"); setFile(null); };
  const pickFile = f => { if (f) { setFile(f); setStep("crop"); } };

  const loadFromUrl = async () => {
    if (!urlInput.trim()) return;
    setBusy("Carregant URL…");
    try {
      const r = await fetch(urlInput.trim());
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const blob = await r.blob();
      if (!blob.type.startsWith("image/")) throw new Error("No és una imatge");
      pickFile(new File([blob], "url.jpg", { type: blob.type }));
      setUrlMode(false); setUrlInput("");
    } catch(e) { alert("No s'ha pogut carregar: " + e.message); }
    finally { setBusy(""); }
  };

  useEffect(() => {
    const onPaste = e => {
      if (step !== "form" || prev) return;
      const img = Array.from(e.clipboardData?.items||[]).find(i=>i.type.startsWith("image/"));
      if (img) { e.preventDefault(); pickFile(img.getAsFile()); }
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [step, prev]);

  const onDragOver  = e => { e.preventDefault(); setDragOver(true); };
  const onDragLeave = () => setDragOver(false);
  const onDrop = e => {
    e.preventDefault(); setDragOver(false);
    const f = Array.from(e.dataTransfer.files).find(f=>f.type.startsWith("image/"));
    if (f) { pickFile(f); return; }
    const url = e.dataTransfer.getData("text/uri-list") || e.dataTransfer.getData("text/plain");
    if (url?.match(/^https?:\/\//)) { setUrlInput(url); setUrlMode(true); }
  };

  const save = async () => {
    if (!name.trim()) return;
    const tagsArr = tags.split(",").map(t=>t.trim()).filter(Boolean);
    if (isEdit && !b64) {
      await onSave({ ...initCell, name: name.trim(), tags: tagsArr }); return;
    }
    if (!b64) return;
    setBusy("Calculant fingerprint…");
    const fingerprint = await computeFingerprint(b64);
    setBusy("Pujant imatge…");
    try {
      const url = await uploadImgbb(b64, name);
      setBusy("Descrivint xapa amb IA…");
      const desc = await describePin(url);
      await onSave({ ...(isEdit ? initCell : {}), name: name.trim(), imageUrl: url, description: desc, fingerprint, tags: tagsArr });
    } catch(e) { alert("Error: " + e.message); }
    finally { setBusy(""); }
  };

  if (step === "camera") return (
    <Modal title={`Càmera${name ? ` — ${name}` : ""}`} onClose={()=>{ stopAllCameras(); setStep("form"); }} maxW={500}>
      <CameraCapture onCapture={f=>{setFile(f);setStep("crop");}} onCancel={()=>{ stopAllCameras(); setStep("form"); }}/>
    </Modal>
  );
  if (step === "crop" && file) return (
    <Modal title="Retallar imatge" onClose={()=>{setStep("form");setFile(null);}} maxW={560}>
      <Cropper file={file} onDone={onCrop} onCancel={()=>{setStep("form");setFile(null);}}/>
    </Modal>
  );

  const canSave = name.trim() && (isEdit || b64);
  const OPTS = [
    { icon:"📷", action: ()=>setStep("camera") },
    { icon:"🖼️", action: ()=>fileRef.current?.click() },
    { icon:"📋", action: async () => {
      try {
        const items = await navigator.clipboard.read();
        for (const item of items) {
          const t = item.types.find(t=>t.startsWith("image/"));
          if (t) { pickFile(new File([await item.getType(t)],"paste.png",{type:t})); return; }
        }
        alert("No hi ha imatge. Fes Ctrl+V.");
      } catch { alert("Fes Ctrl+V directament."); }
    }},
    { icon:"🔗", action: ()=>setUrlMode(v=>!v) },
  ];

  return (
    <Modal title={title} onClose={onClose} maxW={440}>
      {busy
        ? <div style={{textAlign:"center",padding:"32px 0",color:T.muted}}>
            <div style={{fontSize:30,marginBottom:12}}>⏳</div><p>{busy}</p>
          </div>
        : <>
          <label style={LBL}>Nom *</label>
          <input value={name} onChange={e=>setName(e.target.value)}
            placeholder="Ex: CUP Independència…" style={{...INP,marginBottom:12}} autoFocus/>

          <label style={LBL}>Etiquetes <span style={{color:T.muted,fontWeight:400,fontSize:11}}>(separades per comes)</span></label>
          <input value={tags} onChange={e=>setTags(e.target.value)}
            placeholder="vintage, CUP, aniversari…" style={{...INP,marginBottom:12}}/>


          <label style={LBL}>Foto</label>
          {prev
            ? <div style={{display:"flex",alignItems:"center",gap:12,marginBottom:12}}>
                <div style={{position:"relative"}}>
                  <img src={prev} alt="preview" style={{width:80,height:80,objectFit:"cover",borderRadius:10,display:"block"}}/>
                  <button onClick={()=>{setPrev(null);setB64(null);}}
                    style={{position:"absolute",top:-6,right:-6,width:20,height:20,borderRadius:"50%",
                      background:T.danger,border:"none",color:"#fff",cursor:"pointer",fontSize:12,
                      display:"flex",alignItems:"center",justifyContent:"center"}}>×</button>
                </div>
                <span style={{color:T.muted,fontSize:12}}>Foto carregada</span>
              </div>
            : <>
                <div ref={dropRef} onDragOver={onDragOver} onDragLeave={onDragLeave} onDrop={onDrop}
                  style={{border:`2px dashed ${dragOver?T.accent:T.border}`,borderRadius:12,padding:"14px 10px",
                    marginBottom:10,textAlign:"center",transition:"all .12s",
                    background:dragOver?"rgba(247,183,49,0.06)":"transparent"}}>
                  <p style={{color:T.muted,fontSize:11,marginBottom:8}}>
                    {dragOver ? "Deixa anar!" : "Arrossega aqui · Ctrl+V · o tria:"}
                  </p>
                  <div style={{display:"grid",gridTemplateColumns:"repeat(4,1fr)",gap:8}}>
                    {OPTS.map(({icon,action},i)=>(
                      <div key={i} onClick={action}
                        style={{background:T.empty,border:`1px solid ${T.border}`,borderRadius:9,
                          padding:"10px 4px",textAlign:"center",cursor:"pointer",
                          fontSize:18,transition:"border .12s"}}
                        onMouseEnter={e=>e.currentTarget.style.borderColor=T.accent}
                        onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>
                        {icon}
                      </div>
                    ))}
                  </div>
                </div>
                {urlMode && (
                  <div style={{display:"flex",gap:8,marginBottom:10}}>
                    <input value={urlInput} onChange={e=>setUrlInput(e.target.value)}
                      placeholder="https://…" style={{...INP,flex:1}}
                      onKeyDown={e=>e.key==="Enter"&&loadFromUrl()} autoFocus/>
                    <button onClick={loadFromUrl} style={btn("pri",{padding:"10px 14px"})}>→</button>
                  </div>
                )}
              </>
          }
          <input ref={fileRef} type="file" accept="image/*"
            onChange={e=>{pickFile(e.target.files?.[0]);e.target.value="";}} style={{display:"none"}}/>
          <div style={{display:"flex",gap:8,marginTop:8}}>
            <button onClick={onClose} style={btn("def",{flex:1})}>Cancel·lar</button>
            <button onClick={save} disabled={!canSave}
              style={btn("pri",{flex:1,opacity:canSave?1:0.4,cursor:canSave?"pointer":"not-allowed"})}>
              {isEdit ? "Desar ✓" : "Afegir ✓"}
            </button>
          </div>
        </>
      }
    </Modal>
  );
}

// ============================================================
// VIEW XAPA MODAL
// ============================================================
function ViewXapaModal({ cell, idx, cells=[], onClose, onMove, onDelete, onEdit, onNavigate }) {
  const [confirmDel, setConfirmDel] = useState(false);
  const [copied, setCopied] = useState(false);

  // Índexs de les cel·les que tenen xapa (no buides)
  const filledIdxs = cells.map((c,i)=>c?i:-1).filter(i=>i!==-1);
  const pos     = filledIdxs.indexOf(idx);
  const prevIdx = pos > 0                      ? filledIdxs[pos-1] : null;
  const nextIdx = pos < filledIdxs.length - 1  ? filledIdxs[pos+1] : null;

  // Navegació per teclat
  useEffect(() => {
    const onKey = e => {
      if (e.key === "ArrowLeft"  && prevIdx !== null) onNavigate(prevIdx);
      if (e.key === "ArrowRight" && nextIdx !== null) onNavigate(nextIdx);
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [prevIdx, nextIdx]);

  const shareUrl = () => {
    const p = new URLSearchParams(window.location.search);
    if (cell.id) p.set("xapa", cell.id);
    const url = `${window.location.origin}${window.location.pathname}?${p}`;
    navigator.clipboard.writeText(url).then(() => { setCopied(true); setTimeout(()=>setCopied(false),2000); });
  };
  const tags = Array.isArray(cell.tags) ? cell.tags : [];

  const NavBtn = ({label, onClick, disabled}) => (
    <button onClick={onClick} disabled={disabled}
      style={{background:"none",border:`1px solid ${disabled?T.border:T.border}`,
        color:disabled?T.muted:T.text,borderRadius:8,width:36,height:36,fontSize:18,
        cursor:disabled?"default":"pointer",display:"flex",alignItems:"center",
        justifyContent:"center",flexShrink:0,transition:"all .15s"}}
      onMouseEnter={e=>{ if(!disabled) e.currentTarget.style.borderColor=T.accent; }}
      onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>
      {label}
    </button>
  );

  return (
    <Modal onClose={onClose} maxW={440}
      title={
        <div style={{display:"flex",alignItems:"center",gap:8,width:"100%"}}>
          <NavBtn label="‹" onClick={()=>onNavigate(prevIdx)} disabled={prevIdx===null}/>
          <span style={{flex:1,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap",
            fontFamily:"Fraunces",fontSize:18}}>{cell.name}</span>
          <span style={{color:T.muted,fontSize:12,flexShrink:0}}>{pos+1}/{filledIdxs.length}</span>
          <NavBtn label="›" onClick={()=>onNavigate(nextIdx)} disabled={nextIdx===null}/>
        </div>
      }>
      {cell.imageUrl && <img src={cell.imageUrl} alt={cell.name}
        style={{width:"100%",aspectRatio:"1",objectFit:"cover",borderRadius:12,marginBottom:12,display:"block"}}/>}
      {tags.length > 0 && (
        <div style={{display:"flex",gap:6,flexWrap:"wrap",marginBottom:10}}>
          {tags.map((t,i) => (
            <span key={i} style={{background:T.accentBg,color:T.accent,fontSize:11,
              padding:"3px 9px",borderRadius:20,border:`1px solid ${T.accent}44`}}>{t}</span>
          ))}
        </div>
      )}
      {cell.description && <p style={{color:T.muted,fontSize:12,marginBottom:10,lineHeight:1.7,fontStyle:"italic"}}>{cell.description}</p>}
      {confirmDel
        ? <div style={{background:"rgba(238,85,85,0.08)",border:`1px solid ${T.danger}`,borderRadius:10,padding:"12px 14px"}}>
            <p style={{color:T.danger,fontSize:13,marginBottom:10}}>Segur que vols eliminar "{cell.name}"?</p>
            <div style={{display:"flex",gap:8}}>
              <button onClick={()=>setConfirmDel(false)} style={btn("def",{flex:1})}>No</button>
              <button onClick={onDelete} style={btn("red",{flex:1})}>Sí, eliminar</button>
            </div>
          </div>
        : <div style={{display:"grid",gridTemplateColumns:"1fr 1fr 1fr 1fr",gap:8}}>
            <button onClick={shareUrl} style={btn("ghost",{fontSize:12,padding:"8px 4px"})}>
              {copied ? "✓ Copiat" : "🔗 Compartir"}
            </button>
            <button onClick={onEdit} style={btn("ghost",{fontSize:12,padding:"8px 4px"})}>✎ Editar</button>
            <button onClick={onMove} style={btn("ghost",{fontSize:12,padding:"8px 4px"})}>↔ Moure</button>
            <button onClick={()=>setConfirmDel(true)} style={btn("red",{fontSize:12,padding:"8px 4px"})}>🗑</button>
          </div>
      }
    </Modal>
  );
}


// ============================================================
// SEARCH MODAL — algo primer, IA opcional
// ============================================================
function SearchModal({ onClose, getAllXapes, initQuery = "", initImage = null }) {
  const [allXapes] = useState(()=>getAllXapes());
  const [query, setQuery]             = useState(initQuery);
  const [textResults, setTextResults] = useState(()=>{
    if (!initQuery.trim()) return null;
    const lq = initQuery.toLowerCase();
    return getAllXapes().filter(x =>
      x.name?.toLowerCase().includes(lq) || x.description?.toLowerCase().includes(lq) ||
      (Array.isArray(x.tags) && x.tags.some(t => t.toLowerCase().includes(lq)))
    );
  });
  const [imgStep, setImgStep]         = useState(initImage ? "algo" : "idle");
  const [imgFile, setImgFile]         = useState(null);
  const [queryFp, setQueryFp]         = useState(null);
  const [queryB64, setQueryB64]       = useState(initImage);
  const [algoResults, setAlgoResults] = useState(null);
  const [aiResults, setAiResults]     = useState(null);
  const [err, setErr]                 = useState("");
  const fRef = useRef();

  // Si arriba initImage des de URL, executa l'algoritme directament
  useEffect(() => {
    if (!initImage) return;
    (async () => {
      const fp = await computeFingerprint(initImage);
      setQueryFp(fp);
      setAlgoResults(algoFindSimilar(fp, allXapes));
      setImgStep("idle");
    })();
  }, []);

  const doTextSearch = q => {
    setQuery(q); setAlgoResults(null); setAiResults(null); setErr("");
    // Actualitza ?cerca= a la URL (sense pushState per no trencar l'historial)
    const p = new URLSearchParams(window.location.search);
    if (q.trim()) p.set("cerca", q.trim()); else p.delete("cerca");
    window.history.replaceState({}, "", p.toString() ? `?${p}` : window.location.pathname);
    if (!q.trim()) { setTextResults(null); return; }
    const lq = q.toLowerCase();
    setTextResults(allXapes.filter(x =>
      x.name?.toLowerCase().includes(lq) ||
      x.description?.toLowerCase().includes(lq) ||
      (Array.isArray(x.tags) && x.tags.some(t => t.toLowerCase().includes(lq)))
    ));
  };

  const onCrop = async b64 => {
    setQueryB64(b64); setImgStep("algo"); setErr(""); setAiResults(null);
    // Sincronitza ?image= a la URL perquè sigui compartible
    const p = new URLSearchParams(window.location.search);
    p.set("image", encodeURIComponent(b64));
    window.history.replaceState({}, "", `?${p}`);
    const fp = await computeFingerprint(b64);
    setQueryFp(fp);
    const pool = textResults !== null ? textResults : allXapes;
    const results = algoFindSimilar(fp, pool);
    setAlgoResults(results);
    setImgStep("idle");
  };

  const refineWithAI = async () => {
    if (!queryB64 || !algoResults?.length) return;
    setImgStep("ai"); setErr("");
    try {
      const refined = await aiFindSimilar(queryB64, algoResults);
      setAiResults(refined);
    } catch(e) { setErr("Error IA: " + e.message); }
    setImgStep("idle");
  };

  const reset = () => {
    setQuery(""); setTextResults(null); setAlgoResults(null); setAiResults(null);
    setImgFile(null); setQueryB64(null); setQueryFp(null); setErr(""); setImgStep("idle");
    const p = new URLSearchParams(window.location.search);
    p.delete("cerca"); p.delete("image");
    window.history.replaceState({}, "", p.toString() ? `?${p}` : window.location.pathname);
  };

  if (imgStep==="camera") return (
    <Modal title="Càmera — cerca" onClose={()=>{ stopAllCameras(); setImgStep("idle"); }} maxW={500}>
      <CameraCapture onCapture={f=>{setImgFile(f);setImgStep("crop");}} onCancel={()=>{ stopAllCameras(); setImgStep("idle"); }}/>
    </Modal>
  );
  if (imgStep==="crop"&&imgFile) return (
    <Modal title="Retallar imatge de cerca" onClose={()=>{setImgStep("idle");setImgFile(null);}} maxW={560}>
      <Cropper file={imgFile} onDone={onCrop} onCancel={()=>{setImgStep("idle");setImgFile(null);}}/>
    </Modal>
  );

  const showResults = aiResults ?? algoResults;
  const noFp = algoResults !== null && algoResults.length === 0 &&
    (textResults ?? allXapes).some(x => !x.fingerprint);

  return (
    <Modal title="Cerca 🔍" onClose={onClose} maxW={560}>
      {/* TEXT */}
      <label style={LBL}>Busca per nom o descripció</label>
      <div style={{position:"relative",marginBottom:16}}>
        <input value={query} onChange={e=>doTextSearch(e.target.value)}
          placeholder="Ex: vermell, CUP, gat…" style={{...INP,paddingRight:36}} autoFocus/>
        {query&&<button onClick={()=>doTextSearch("")}
          style={{position:"absolute",right:10,top:"50%",transform:"translateY(-50%)",background:"none",border:"none",color:T.muted,cursor:"pointer",fontSize:16}}>×</button>}
      </div>

      <div style={{display:"flex",alignItems:"center",gap:10,marginBottom:14}}>
        <div style={{flex:1,height:1,background:T.border}}/>
        <span style={{color:T.muted,fontSize:12}}>{textResults!==null?`refinar ${textResults.length} resultats`:"o cerca"} per imatge</span>
        <div style={{flex:1,height:1,background:T.border}}/>
      </div>

      {/* IMAGE INPUT */}
      {imgStep==="algo"
        ?<div style={{textAlign:"center",padding:"20px 0",color:T.muted,marginBottom:14}}>
          <div style={{fontSize:26,marginBottom:8}}>⚡</div>
          <p>Analitzant imatge…</p>
        </div>
        :imgStep==="ai"
        ?<div style={{textAlign:"center",padding:"20px 0",color:T.muted,marginBottom:14}}>
          <div style={{fontSize:26,marginBottom:8}}>🤖</div>
          <p>L'IA refina els resultats…</p>
        </div>
        :<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginBottom:14}}>
          {[{icon:"📷",label:"Càmera",action:()=>setImgStep("camera")},
            {icon:"🖼️",label:"Fitxer",action:()=>fRef.current?.click()}
          ].map(({icon,label,action})=>(
            <div key={label} onClick={action}
              style={{border:`2px dashed ${T.border}`,borderRadius:12,padding:"18px 10px",textAlign:"center",cursor:"pointer",transition:"border .12s"}}
              onMouseEnter={e=>e.currentTarget.style.borderColor=T.accent}
              onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>
              <div style={{fontSize:22,marginBottom:5}}>{icon}</div>
              <p style={{color:T.muted,fontSize:12}}>{label}</p>
            </div>
          ))}
        </div>
      }
      <input ref={fRef} type="file" accept="image/*"
        onChange={e=>{const f=e.target.files?.[0];if(f){setImgFile(f);setImgStep("crop");e.target.value="";}}}
        style={{display:"none"}}/>

      {err&&<p style={{color:T.danger,fontSize:13,marginBottom:12}}>{err}</p>}

      {/* RESULTS */}
      {showResults!==null&&<>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:10}}>
          <p style={{color:T.muted,fontSize:12}}>
            {aiResults ? "✨ Refinat per IA" : "⚡ Resultats algorítmics"} · {showResults.length} xapes
          </p>
          {algoResults!==null&&algoResults.length>0&&!aiResults&&imgStep==="idle"&&(
            <button onClick={refineWithAI}
              style={btn("ghost",{fontSize:12,padding:"5px 12px"})}>
              Refinar amb IA 🤖
            </button>
          )}
          {aiResults&&(
            <button onClick={()=>setAiResults(null)}
              style={{background:"none",border:"none",color:T.muted,fontSize:12,cursor:"pointer"}}>
              ← Tornar a algo
            </button>
          )}
        </div>

        {noFp&&<p style={{color:T.muted,fontSize:11,marginBottom:10}}>
          ⚠️ Algunes xapes antigues no tenen fingerprint — torna a afegir-les per incloure-les.
        </p>}

        {showResults.length===0
          ?<div style={{textAlign:"center",padding:"24px 0",color:T.muted}}>Cap resultat 😔</div>
          :<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(105px,1fr))",gap:9,maxHeight:320,overflowY:"auto"}}>
            {showResults.map((x,i)=>(
              <div key={i} style={{background:T.empty,borderRadius:9,overflow:"hidden",border:`1px solid ${T.border}`,position:"relative"}}>
                {x.imageUrl&&<img src={x.imageUrl} alt={x.name} style={{width:"100%",aspectRatio:"1",objectFit:"cover",display:"block"}}/>}
                {x.score!==undefined&&(
                  <div style={{position:"absolute",top:4,right:4,background:"rgba(0,0,0,0.7)",borderRadius:4,padding:"1px 5px",fontSize:10,color:T.accent}}>
                    {Math.round(x.score*100)}%
                  </div>
                )}
                <div style={{padding:"6px 8px"}}>
                  <p style={{fontSize:11,fontWeight:500,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{x.name}</p>
                </div>
              </div>
            ))}
          </div>
        }
        <button onClick={reset} style={btn("def",{width:"100%",marginTop:12,fontSize:13})}>Netejar cerca</button>
      </>}

      {/* text-only results (no image search done yet) */}
      {textResults!==null&&algoResults===null&&<>
        <p style={{color:T.muted,fontSize:12,marginBottom:10}}>{textResults.length} resultats per "{query}"</p>
        {textResults.length===0
          ?<div style={{textAlign:"center",padding:"20px 0",color:T.muted}}>Cap resultat 😔</div>
          :<div style={{display:"grid",gridTemplateColumns:"repeat(auto-fill,minmax(105px,1fr))",gap:9,maxHeight:300,overflowY:"auto"}}>
            {textResults.map((x,i)=>(
              <div key={i} style={{background:T.empty,borderRadius:9,overflow:"hidden",border:`1px solid ${T.border}`}}>
                {x.imageUrl&&<img src={x.imageUrl} alt={x.name} style={{width:"100%",aspectRatio:"1",objectFit:"cover",display:"block"}}/>}
                <div style={{padding:"6px 8px"}}>
                  <p style={{fontSize:11,fontWeight:500,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{x.name}</p>
                </div>
              </div>
            ))}
          </div>
        }
      </>}
    </Modal>
  );
}