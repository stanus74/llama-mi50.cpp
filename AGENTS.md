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

### 2️⃣ Lies `patches_komplett/cuda_ohne_gfx906.diff`

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

1. Schau `patches_komplett/cuda_ohne_gfx906.diff` → Kategorie X
2. Lies welche Dateien betroffen sind
3. Generiere Patch: `git diff ... > patch.diff`
4. Wende an: `git apply patch.diff`
5. Falls Fehler: `git apply --reject` + manuelles Mergen
6. Build + Test
7. Commit wenn erfolgreich

---
