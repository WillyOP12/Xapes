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
// CROPPER
// ============================================================
function Cropper({ file, onDone, onCancel }) {
  const cvRef = useRef(); const imgRef = useRef();
  const [rdy, setRdy] = useState(false);
  const [crop, setCrop] = useState(null);
  const drag = useRef(null);

  useEffect(() => {
    const u = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      imgRef.current = img;
      const s = Math.min(img.width, img.height);
      setCrop({ x:(img.width-s)/2, y:(img.height-s)/2, s });
      setRdy(true);
    };
    img.src = u;
    return () => URL.revokeObjectURL(u);
  }, [file]);

  useEffect(() => {
    if (!rdy || !crop || !cvRef.current) return;
    const cv = cvRef.current, img = imgRef.current, ctx = cv.getContext("2d");
    const mw = Math.min(460, (window.innerWidth||460)-48);
    const sc = mw/img.width;
    cv.width = img.width*sc; cv.height = img.height*sc;
    ctx.drawImage(img, 0, 0, cv.width, cv.height);
    ctx.fillStyle="rgba(0,0,0,0.55)"; ctx.fillRect(0,0,cv.width,cv.height);
    const [cx,cy,cs]=[crop.x*sc,crop.y*sc,crop.s*sc];
    ctx.drawImage(img, crop.x,crop.y,crop.s,crop.s, cx,cy,cs,cs);
    ctx.strokeStyle=T.accent; ctx.lineWidth=2; ctx.strokeRect(cx,cy,cs,cs);
    const hs=10; ctx.fillStyle=T.accent;
    [[cx,cy],[cx+cs-hs,cy],[cx,cy+cs-hs],[cx+cs-hs,cy+cs-hs]].forEach(([hx,hy])=>ctx.fillRect(hx,hy,hs,hs));
  }, [rdy, crop]);

  const toImg = e => {
    const cv=cvRef.current, r=cv.getBoundingClientRect(), img=imgRef.current;
    const sc=img.width/cv.clientWidth, s=e.touches?e.touches[0]:e;
    return { x:(s.clientX-r.left)*sc, y:(s.clientY-r.top)*sc };
  };
  const dn = e => {
    if (!crop) return; e.preventDefault();
    const p=toImg(e);
    if (p.x>=crop.x&&p.x<=crop.x+crop.s&&p.y>=crop.y&&p.y<=crop.y+crop.s)
      drag.current={ox:p.x-crop.x,oy:p.y-crop.y};
  };
  const mv = e => {
    if (!drag.current) return; e.preventDefault();
    const p=toImg(e), img=imgRef.current;
    setCrop(c=>({...c, x:Math.max(0,Math.min(img.width-c.s,p.x-drag.current.ox)), y:Math.max(0,Math.min(img.height-c.s,p.y-drag.current.oy))}));
  };
  const up = () => { drag.current=null; };
  const confirm = () => {
    const oc=document.createElement("canvas"); oc.width=oc.height=400;
    oc.getContext("2d").drawImage(imgRef.current,crop.x,crop.y,crop.s,crop.s,0,0,400,400);
    onDone(oc.toDataURL("image/jpeg",0.85).split(",")[1]);
  };

  if (!rdy) return <div style={{color:T.muted,textAlign:"center",padding:24}}>Carregant…</div>;
  return (
    <div>
      <p style={{color:T.muted,fontSize:13,marginBottom:10}}>Arrossega per posicionar · ajusta la mida amb el control</p>
      <canvas ref={cvRef} style={{width:"100%",cursor:"grab",touchAction:"none",borderRadius:8,display:"block"}}
        onMouseDown={dn} onMouseMove={mv} onMouseUp={up} onMouseLeave={up}
        onTouchStart={dn} onTouchMove={mv} onTouchEnd={up}/>
      {imgRef.current&&crop&&(
        <input type="range" min={30} max={Math.min(imgRef.current.width,imgRef.current.height)} value={crop.s}
          onChange={e=>{const ns=+e.target.value;setCrop(c=>({s:ns,x:Math.min(c.x,imgRef.current.width-ns),y:Math.min(c.y,imgRef.current.height-ns)}));}}
          style={{width:"100%",margin:"12px 0 4px",accentColor:T.accent}}/>
      )}
      <div style={{display:"flex",gap:8,marginTop:10}}>
        <button onClick={onCancel} style={btn("def",{flex:1})}>Cancel·lar</button>
        <button onClick={confirm} style={btn("pri",{flex:1})}>Retallar ✓</button>
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
  return (
    <div style={{position:"fixed",inset:0,background:"rgba(0,0,0,0.88)",display:"flex",alignItems:"center",justifyContent:"center",zIndex:999,padding:16}}
      onClick={e=>e.target===e.currentTarget&&onClose()}>
      <div style={{background:T.card,borderRadius:16,padding:24,width:"100%",maxWidth:maxW,maxHeight:"92vh",overflowY:"auto",border:`1px solid ${T.border}`}}>
        <div style={{display:"flex",justifyContent:"space-between",alignItems:"center",marginBottom:20}}>
          <h3 style={{fontFamily:"Fraunces",fontSize:20,color:T.text}}>{title}</h3>
          <button onClick={onClose} style={{background:"none",border:"none",color:T.muted,fontSize:22,cursor:"pointer",lineHeight:1,padding:"0 2px"}}>×</button>
        </div>
        {children}
      </div>
    </div>
  );
}

// ============================================================
// APP ROOT
// ============================================================
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

  useEffect(() => {
    (async () => {
      const a = await S.get("xapes-albums");
      setAlbums(Array.isArray(a) ? a : []);
      setLoading(false);
    })();
  }, []);

  const saveAlbums = async a => { setAlbums(a); await S.set("xapes-albums", a); };
  const saveData   = async (aid, d) => { setData(d); await S.set(`xapes-ad-${aid}`, d); };

  const openAlbum = async a => {
    const raw = await S.get(`xapes-ad-${a.id}`);
    const d = (raw && typeof raw === "object" && !Array.isArray(raw))
      ? raw
      : { sheets: [], bigItems: [] };
    if (!Array.isArray(d.sheets))   d.sheets   = [];
    if (!Array.isArray(d.bigItems)) d.bigItems = [];
    setAlbum(a); setData(d); setView("album");
  };

  const goBack = () => {
    if (view==="sheet") { setView("album"); setMv(null); }
    else { setView("home"); setAlbum(null); setData(null); }
  };

  const curSheet  = () => data?.sheets?.[sheetIdx];
  const openSheet = i  => { setSheetIdx(i); setView("sheet"); };
  const goPrev    = () => { const i=Math.max(0,sheetIdx-1); setSheetIdx(i); };
  const goNext    = () => { const i=Math.min((data?.sheets?.length||1)-1,sheetIdx+1); setSheetIdx(i); };

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
          onSave={async({name,imageUrl,description})=>{
            const nd={...data,sheets:data.sheets.map((s,si)=>si!==sheetIdx?s:(()=>{const c=[...s.cells];c[modal.idx]={id:uid(),name,imageUrl,description};return{...s,cells:c};})())};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="view"&&(
        <ViewXapaModal cell={modal.cell} onClose={()=>setModal(null)}
          onMove={()=>{setMv({type:"cell",idx:modal.idx,item:modal.cell});setModal(null);}}
          onDelete={()=>{
            const nd={...data,sheets:data.sheets.map((s,si)=>si!==sheetIdx?s:(()=>{const c=[...s.cells];c[modal.idx]=null;return{...s,cells:c};})())};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="addBig"&&(
        <AddXapaModal title="Afegir xapa gran" busy={busy} setBusy={setBusy} onClose={()=>setModal(null)}
          onSave={async({name,imageUrl,description})=>{
            const nd={...data,bigItems:[...(data.bigItems||[]),{id:uid(),name,imageUrl,description}]};
            saveData(album.id,nd); setModal(null);
          }}/>
      )}
      {modal?.t==="viewBig"&&(
        <ViewXapaModal cell={modal.item} onClose={()=>setModal(null)}
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
      {searchOpen&&<SearchModal onClose={()=>setSearchOpen(false)} getAllXapes={allXapes}/>}
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
function AddXapaModal({ title, busy, setBusy, onClose, onSave }) {
  const [name, setName]   = useState("");
  const [step, setStep]   = useState("form"); // form | camera | crop
  const [file, setFile]   = useState(null);
  const [b64, setB64]     = useState(null);
  const [prev, setPrev]   = useState(null);
  const fileRef = useRef();

  const onCrop = d => { setB64(d); setPrev(`data:image/jpeg;base64,${d}`); setStep("form"); setFile(null); };
  const pickFile = f => { if(f){ setFile(f); setStep("crop"); } };

  const save = async () => {
    if (!name.trim()||!b64) return;
    setBusy("Calculant fingerprint…");
    const fingerprint = await computeFingerprint(b64);
    setBusy("Pujant imatge…");
    try {
      const url = await uploadImgbb(b64, name);
      setBusy("Descrivint xapa amb IA…");
      const desc = await describePin(url);
      await onSave({name:name.trim(), imageUrl:url, description:desc, fingerprint});
    } catch(e) { alert("Error: "+e.message); }
    finally { setBusy(""); }
  };

  if (step==="camera") return (
    <Modal title="Càmera" onClose={()=>{ stopAllCameras(); setStep("form"); }} maxW={500}>
      <CameraCapture onCapture={f=>{setFile(f);setStep("crop");}} onCancel={()=>{ stopAllCameras(); setStep("form"); }}/>
    </Modal>
  );
  if (step==="crop"&&file) return (
    <Modal title="Retallar imatge" onClose={()=>{setStep("form");setFile(null);}}>
      <Cropper file={file} onDone={onCrop} onCancel={()=>{setStep("form");setFile(null);}}/>
    </Modal>
  );

  return (
    <Modal title={title} onClose={onClose}>
      {busy
        ?<div style={{textAlign:"center",padding:"32px 0",color:T.muted}}>
          <div style={{fontSize:30,marginBottom:12}}>⏳</div>
          <p>{busy}</p>
        </div>
        :<>
          <label style={LBL}>Nom de la xapa</label>
          <input value={name} onChange={e=>setName(e.target.value)} placeholder="Ex: Polzets, CUP, Arquet…"
            style={{...INP,marginBottom:16}} autoFocus/>
          <label style={LBL}>Foto</label>
          {prev
            ?<div style={{marginBottom:16}}>
              <img src={prev} alt="preview" style={{width:100,height:100,objectFit:"cover",borderRadius:10,display:"block",marginBottom:8}}/>
              <button onClick={()=>{setPrev(null);setB64(null);}} style={btn("def",{fontSize:12,padding:"6px 12px"})}>Canviar foto</button>
            </div>
            :<div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10,marginBottom:16}}>
              <div onClick={()=>setStep("camera")}
                style={{border:`2px dashed ${T.border}`,borderRadius:12,padding:"22px 10px",textAlign:"center",cursor:"pointer",transition:"border .12s"}}
                onMouseEnter={e=>e.currentTarget.style.borderColor=T.accent}
                onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>
                <div style={{fontSize:26,marginBottom:6}}>📷</div>
                <p style={{color:T.muted,fontSize:12}}>Càmera</p>
              </div>
              <div onClick={()=>fileRef.current?.click()}
                style={{border:`2px dashed ${T.border}`,borderRadius:12,padding:"22px 10px",textAlign:"center",cursor:"pointer",transition:"border .12s"}}
                onMouseEnter={e=>e.currentTarget.style.borderColor=T.accent}
                onMouseLeave={e=>e.currentTarget.style.borderColor=T.border}>
                <div style={{fontSize:26,marginBottom:6}}>🖼️</div>
                <p style={{color:T.muted,fontSize:12}}>Fitxer</p>
              </div>
            </div>
          }
          <input ref={fileRef} type="file" accept="image/*"
            onChange={e=>{pickFile(e.target.files?.[0]);e.target.value="";}}
            style={{display:"none"}}/>
          <div style={{display:"flex",gap:8,marginTop:4}}>
            <button onClick={onClose} style={btn("def",{flex:1})}>Cancel·lar</button>
            <button onClick={save} disabled={!name.trim()||!b64}
              style={btn("pri",{flex:1,opacity:(!name.trim()||!b64)?0.4:1,cursor:(!name.trim()||!b64)?"not-allowed":"pointer"})}>
              Afegir ✓
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
function ViewXapaModal({ cell, onClose, onMove, onDelete }) {
  return (
    <Modal title={cell.name} onClose={onClose}>
      {cell.imageUrl&&<img src={cell.imageUrl} alt={cell.name} style={{width:"100%",aspectRatio:"1",objectFit:"cover",borderRadius:12,marginBottom:14,display:"block"}}/>}
      {cell.description&&<p style={{color:T.muted,fontSize:13,marginBottom:20,lineHeight:1.7}}>{cell.description}</p>}
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:10}}>
        <button onClick={onMove} style={btn("ghost")}>↔ Moure</button>
        <button onClick={onDelete} style={btn("red")}>🗑 Eliminar</button>
      </div>
    </Modal>
  );
}

// ============================================================
// SEARCH MODAL — algo primer, IA opcional
// ============================================================
function SearchModal({ onClose, getAllXapes }) {
  const [allXapes] = useState(()=>getAllXapes());
  const [query, setQuery]             = useState("");
  const [textResults, setTextResults] = useState(null);
  const [imgStep, setImgStep]         = useState("idle"); // idle | camera | crop | algo | ai
  const [imgFile, setImgFile]         = useState(null);
  const [queryFp, setQueryFp]         = useState(null);
  const [queryB64, setQueryB64]       = useState(null);
  const [algoResults, setAlgoResults] = useState(null);
  const [aiResults, setAiResults]     = useState(null);
  const [err, setErr]                 = useState("");
  const fRef = useRef();

  const doTextSearch = q => {
    setQuery(q); setAlgoResults(null); setAiResults(null); setErr("");
    if (!q.trim()) { setTextResults(null); return; }
    const lq = q.toLowerCase();
    setTextResults(allXapes.filter(x =>
      x.name?.toLowerCase().includes(lq) || x.description?.toLowerCase().includes(lq)
    ));
  };

  const onCrop = async b64 => {
    setQueryB64(b64); setImgStep("algo"); setErr(""); setAiResults(null);
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