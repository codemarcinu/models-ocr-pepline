# ✅ CHECKLIST: Perplexica Setup

## 📋 PRE-FLIGHT

- [ ] WSL2 zainstalowany
- [ ] Windows 11 + Ubuntu 22.04
- [ ] RTX 3060 dostępna
- [ ] 100GB dysku SSD free
- [ ] Internet ≥50 Mbps

## 🔧 INSTALACJA

### FAZA 1: Środowisko (20 min)
- [ ] `wsl --list --verbose` pokazuje Ubuntu v2
- [ ] `node --version` ≥18.0.0
- [ ] `npm --version` ≥9.0.0
- [ ] Folder `~/projects/perplexica` stworzony

### FAZA 2: Perplexica (30 min)
- [ ] `git clone` bez błędów
- [ ] `npm install` ukończony
- [ ] `npm run build` bez błędów
- [ ] Folder `.next/` istnieje

### FAZA 3: Bing API (20 min)
- [ ] Konto Azure.com stworzony
- [ ] "Bing Search API v7" resource created
- [ ] Pricing tier: FREE (7 req/sec)
- [ ] KEY 1 skopiowany
- [ ] `.env.local` stworzony z KEY
- [ ] `.gitignore` zawiera `.env.local`

### FAZA 4: Ollama (40 min)
- [ ] `ollama serve` uruchomiony (Terminal 1)
- [ ] `curl http://localhost:11434/api/tags` działa
- [ ] `ollama pull mistral` pobrano (4.1 GB)
- [ ] `ollama list` pokazuje mistral
- [ ] Test: `ollama run mistral "test"` działa

### FAZA 5: Test (40 min)
- [ ] `npm start` bez błędów (Terminal 2)
- [ ] `http://localhost:3000` otwiera się
- [ ] UI Perplexica widoczny
- [ ] Wyszukiwanie "Python" zwraca wyniki
- [ ] Czas odpowiedzi <6 sekund
- [ ] GPU Utilization 60-75% (podczas synthesis)

## 🎯 VERIFICATION

- [ ] Web search działa (Bing API)
- [ ] LLM synthesis działa (Mistral)
- [ ] Sources linkowane poprawnie
- [ ] Brak error 500
- [ ] Brak "Connection refused"
- [ ] GPU RAM <9GB usage

## 🔒 SECURITY

- [ ] `.env.local` ma chmod 600
- [ ] `.env.local` w `.gitignore`
- [ ] `git status` nie pokazuje .env
- [ ] API key nie commitowany

## 📊 MONITORING

- [ ] `nvidia-smi -l 1` pokazuje GPU load
- [ ] Token/sec widoczny (5-10)
- [ ] Web search: 2-3s
- [ ] LLM: 2-4s
- [ ] Total: 4-6s

## ✅ PRODUCTION READY

- [ ] Wszystkie FAZY ukończone
- [ ] Wszystkie VERIFICATION checkpointy przeszły
- [ ] SECURITY hardening done
- [ ] MONITORING setup
- [ ] Performance acceptable

---

## 🚀 READY TO USE

Jeśli wszystkie [ ] zaznaczone:
✅ System gotowy do użycia!

Codziennie:
```bash
# Terminal 1
ollama serve

# Terminal 2
cd ~/projects/perplexica && npm start

# Browser
http://localhost:3000
```

---

## 🆘 TROUBLESHOOTING QUICK LINKS

- Port 3000 occupied? → `kill -9 $(lsof -ti:3000)`
- Ollama not responding? → Check `ollama serve` in Terminal 1
- Bing API error? → Check `.env.local` has valid KEY
- Model missing? → `ollama pull mistral`
- npm install fails? → `npm cache clean --force && npm install`

---

Status: Ready for production
Date: 17 January 2026