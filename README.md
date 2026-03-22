# Xapes 📌

Gestor de col·leccions de xapes amb IA.

## Requisits

- Node.js 18+
- Compte a [ImgBB](https://imgbb.com) → API key a [api.imgbb.com](https://api.imgbb.com)
- Compte a [Groq](https://console.groq.com) → API key gratuïta

## Desenvolupament local

```bash
npm install
cp .env.example .env
# Edita .env i posa les teves claus
npm run dev
```

## Deploy a Vercel

1. Puja el repo a GitHub
2. Importa el projecte a [vercel.com](https://vercel.com)
3. A **Settings → Environment Variables** afegeix:
   - `VITE_GROQ_API` → la teva clau de Groq
   - `VITE_IMGBB_API` → la teva clau d'ImgBB
4. Deploy!

## Variables d'entorn

| Variable | Descripció |
|----------|------------|
| `VITE_GROQ_API` | API key de Groq (visió per IA) |
| `VITE_IMGBB_API` | API key d'ImgBB (allotjament d'imatges) |

## Funcionalitats

- 📁 Àlbums amb nom i color personalitzable
- 📄 Fulles 6×8 navegables com slides
- 📌 Xapes amb foto (càmera o fitxer), nom i descripció automàtica per IA
- 📦 Secció de xapes grans (fora de graella)
- 🔍 Cerca per text i per imatge dins l'àlbum actiu
- ↔️ Mou xapes entre caselles
