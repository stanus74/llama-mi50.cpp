# 🤖 AGENTS.md — Anleitung für AI-Assistenten

## 📋 KONTEXT VERSTEHEN

Bevor du eine Aufgabe bearbeitest, **musst du den Projekthintergrund verstehen**:

### 1️⃣ Lies `plans/Optimierung.md`

Diese Datei enthält:
- **Ziel des Projekts:** gfx906 (AMD MI50/MI60) Optimierungen für llama.cpp
- **Was muss gemacht werden:** Welche Patches/Features sind prioritär
- **Status:** Was ist bereits done, was nicht
- **Performance-Ziele:** Wie viel Speedup wird erwartet

**Nimm dir Zeit für diese Datei!** Sie ist die Roadmap für alles.

---

### 2️⃣ Lies `plans/Diffs-gfx906-mainline.md`

Diese Datei dokumentiert:
- **Alle Unterschiede** zwischen gfx906-Fork und Mainline (llama.cpp b7871)
- **Kategorisierte Optimierungen** (Kategorie A-H):
  - A: Neue gfx906 Kernel-Dateien
  - B: Kernel-Dispatch/Registrierung
  - C: Flash-Attention Optimierungen
  - D: Quantization Pipeline
  - E: Wave64/MoE Fix
  - F: DPP Warp Utils
  - G: Build-System
  - H: Scripts & Tools

- **Welche Dateien geändert sind:**
  - `ggml/src/ggml-cuda/common.cuh`
  - `ggml/src/ggml-cuda/fattn*.cu*`
  - `ggml/src/ggml-cuda/mmq.cu`
  - `ggml/src/ggml-cuda/mmvq.cu`
  - `ggml/src/ggml-cuda/gfx906/` (neue Dateien)
  - `ggml/src/ggml-cuda/CMakeLists.txt`

- **Detaillierte Code-Änderungen** für jede Kategorie

**Nutze diese Datei als Referenz beim Mergen/Patchen!**

---

## 🎯 ARBEITSABLAUF

### Schritt 1: Kontext lesen (5 min)
```
1. Öffne plans/Optimierung.md
2. Verstehe: Was ist das Problem? Was ist die Lösung?
3. Öffne plans/Diffs-gfx906-mainline.md
4. Verstehe: Welche Kategorien sind relevant für DEINE Aufgabe?
```

### Schritt 2: Aufgabe verstehen (2 min)
```
1. Lies die konkrete Aufgabe (z.B. "Patch Kategorie C anwenden")
2. Finde die relevanten Kategorien in Diffs-Datei
3. Schau dir die Dateien an, die geändert werden
```

### Schritt 3: Implementierung (variabel)
```
1. Mache die Änderungen
2. Teste nach jedem Schritt (Build + Benchmark)
3. Dokumentiere was du gemacht hast
```

### Schritt 4: Validierung (5 min)
```
1. Prüfe dass Build erfolgreich ist
2. Vergleiche Performance (eval time vorher/nachher)
3. Commit mit aussagekräftiger Message
```

---

## 💡 TIPPS FÜR DIESE AUFGABEN

### ✅ DO's

- **Lese ALLE Kontextdateien** bevor du fragst
- **Teste nach jedem Patch** — Build + Benchmark
- **Committe nach erfolgreichen Tests** (nicht vorher!)
- **Nutze git branches** — nicht direkt in master arbeiten
- **Sei systematisch** — eine Kategorie nach der anderen
- **Dokumentiere Benchmarks** — eval time vorher/nachher

### ❌ DON'Ts

- **Blinde Patches anwenden** ohne zu verstehen was sie tun
- **Ganze Forks integrieren** ohne zu testen
- **Struktur-Annahmen machen** — prüfe ob Datei/Struct noch existiert
- **Build-Fehler ignorieren** — diagnostic immer sofort
- **Performance-Vergleiche vergessen** — das ist das Ziel!

---

## 🔧 HÄUFIGE SZENARIEN

### Szenario A: "Patch Kategorie X anwenden"

1. Schau `plans/Diffs-gfx906-mainline.md` → Kategorie X
2. Lies welche Dateien betroffen sind
3. Generiere Patch: `git diff ... > patch.diff`
4. Wende an: `git apply patch.diff`
5. Falls Fehler: `git apply --reject` + manuelles Mergen
6. Build + Test
7. Commit wenn erfolgreich

---

### Szenario B: "Integration von Fork X vs Mainline Y"

1. Lies `plans/Optimierung.md` für Hintergrund
2. Vergleiche: `diff -r fork/ggml mainline/ggml`
3. Kategorisiere Unterschiede (nutze `plans/Diffs-gfx906-mainline.md`)
4. Extrahiere Patches pro Kategorie
5. Wende nacheinander an mit Tests dazwischen
6. Dokumentiere Performance-Ergebnisse

---

### Szenario C: "Fehler bei Build/Integration"

1. Lese Fehlermeldung genau
2. Schau `plans/Diffs-gfx906-mainline.md` ob ähnliche Probleme dokumentiert sind
3. Prüfe ob Struktur-Namen sich geändert haben (sehr häufig!)
4. Manuelles Mergen in VS Code
5. Build + Test erneut

---

## 📊 WICHTIGE DATEIEN & VERZEICHNISSE

```
/opt/llama-mi50.cpp/
├── /plans/                          ← CONTEXT LESEN!
│   ├── Optimierung.md              ← Projekt-Ziele
│   └── Diffs-gfx906-mainline.md    ← Alle Unterschiede
├── ggml/src/ggml-cuda/
│   ├── common.cuh                  ← DPP Utils (F)
│   ├── fattn*.cu*                  ← Flash-Attn (C)
│   ├── mmq.cu / mmvq.cu            ← Dispatcher (B)
│   ├── CMakeLists.txt              ← Build (G)
│   └── gfx906/                     ← Neue Kernel (A)
│       ├── gfx906-common.cuh
│       ├── gfx906-config.h
│       ├── gfx906-fattn-q8.*
│       ├── gfx906-mmvq*.cuh
│       ├── gfx906-mmq*.cuh
│       └── template-instances/
├── build/                          ← Build-Artefakte
└── .git/                           ← Git-History

plans/patch          ← Patch-Dateien
├── C-flash-attn-gfx906.patch
├── F-dpp-warp-utils.patch
└── ... (weitere Patches)
```

---

## 🎬 WORKFLOW BEISPIEL

**Aufgabe: "Integriere Kategorie C (Flash-Attention) aus iacopPBK-Fork"**

```bash
# 1. CONTEXT VERSTEHEN
cat plans/Optimierung.md
cat plans/Diffs-gfx906-mainline.md | grep -A 50 "Kategorie C"

# 2. PATCH EXTRAHIEREN
diff -u /tmp/llama-mainline/ggml/src/ggml-cuda/fattn.cu \
        /tmp/llama-iacopbk/ggml/src/ggml-cuda/fattn.cu \
  > /tmp/cat_c.patch

# 3. BRANCH ERSTELLEN
git checkout -b cat-c-integration

# 4. PATCH ANWENDEN
git apply /tmp/cat_c.patch  # oder --reject falls Fehler

# 5. BUILD + TEST
rm -rf build && mkdir build && cd build
cmake .. -DGGML_HIP=ON -DGGML_HIPBLAS=ON
make -j$(nproc) 2>&1 | grep error

# 6. BENCHMARK
./bin/llama-bench -m <model> -ngl 999 -flash-attn on | tee /tmp/bench_after_c.txt

# 7. DOKUMENTIERE ERGEBNIS
# "Flash-Attention Kategorie C: 12% Performance-Boost"
# eval time: 120ms → 106ms

# 8. COMMIT
git add -A
git commit -m "Add: Flash-Attention gfx906 optimizations (Category C)

- Integrate fattn optimizations from iacopPBK fork
- Improves TG/PP performance on MI50/MI60
- Performance: +12% (eval time 120ms → 106ms)"
```

---

## ⚠️ HÄUFIGE PROBLEME & LÖSUNGEN

| Problem | Ursache | Lösung |
|---------|---------|--------|
| `unknown type 'ggml_cuda_graph_node_properties'` | Struct-Name geändert in Mainline | Prüfe aktuelle Definition in Mainline, adapter den Code |
| `git apply` schlägt fehl | Basis-Commit unterschiedlich | Nutze `--reject`, dann manuell mergen |
| Build dauert sehr lange | Template-Instances kompilieren | Normal, ist kein Fehler |
| Performance gleich/schlechter | Patch passt nicht zu Mainline | Prüfe ob Änderungen bereits enthalten sind |
| Binary größer | gfx906-Kernels hinzugefügt | Erwartet, ist okay |

---

## 📝 TEMPLATE FÜR AUFGABEN-DOKUMENTATION

```markdown
## Aufgabe: [Name]

### Kontext
- Ziel: [Was wird gemacht]
- Kategorie: [A/B/C/D/E/F/G/H]
- Betroffene Dateien: [ggml/src/ggml-cuda/...]

### Plan
1. [ ] Patch generieren
2. [ ] Branch erstellen
3. [ ] Patch anwenden
4. [ ] Commit Beschreibung erstellen

### Ergebnis
- Build: ✅ Erfolgreich / ❌ Fehler
- Performance: [vorher] → [nachher] ([+X%])
- Commit: [Hash]

### Anmerkungen
[Was hat geklappt, was nicht]
```

---

## 🚀 SCHNELL-CHECKLISTE VOR JEDER AUFGABE

- [ ] Ich habe `plans/Optimierung.md` gelesen
- [ ] Ich habe `plans/Diffs-gfx906-mainline.md` für relevante Kategorie gelesen
- [ ] Ich verstehe welche Dateien geändert werden
- [ ] Ich erstelle einen neuen Branch (nicht in master arbeiten)
- [ ] Ich teste nach jedem Patch (Build + Benchmark)
- [ ] Ich dokumentiere Performance-Ergebnisse
- [ ] Ich committe mit aussagekräftiger Message

---

## 🎯 ZUSAMMENFASSUNG

**Vor jeder Aufgabe:**
1. 📖 Lese `plans/Optimierung.md` — verstehe das Projekt
2. 📖 Lese `plans/Diffs-gfx906-mainline.md` — verstehe die Änderungen
3. 🔧 Identifiziere betroffene Dateien & Kategorien
4. 🧪 Teste nach jedem Patch
5. 📊 Dokumentiere Performance-Ergebnisse
6. 💾 Committe mit klarer Message

**Sprache: Immer Deutsch** 

