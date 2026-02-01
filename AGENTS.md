# 🤖 AGENTS.md — Anleitung für AI-Assistenten

## 📋 KONTEXT VERSTEHEN

Bevor du eine Aufgabe bearbeitest, **musst du den Projekthintergrund verstehen**:

### 1️⃣ Lies `plans/Optimierung.md`

Diese Datei enthält:
- **Ziel des Projekts:** gfx906 (AMD MI50/MI60) Optimierungen für llama.cpp
- **Was muss gemacht werden:** Welche Patches/Features sind prioritär
- **Status:** Was ist bereits done, was nicht


**Nimm dir Zeit für diese Datei!** Sie ist die Roadmap für alles.

---

### 2️⃣ Lies `patches_komplett/cuda_ohne_gfx906.diff`

Diese Datei dokumentiert:
- **Alle Unterschiede** zwischen gfx906-Fork und Mainline (llama.cpp b7871)
- **Kategorisierte Optimierungen** (Kategorie A-H):
  - A: Neue gfx906 Kernel-Dateien
    - `ggml/src/ggml-cuda/gfx906/` (neue Dateien)
  - B: Kernel-Dispatch/Registrierung
  - C: Flash-Attention Optimierungen
    - `ggml/src/ggml-cuda/fattn*.cu*`
  - D: Quantization Pipeline
    - `ggml/src/ggml-cuda/mmq.cu`
    - `ggml/src/ggml-cuda/mmq.cuh`
    - `ggml/src/ggml-cuda/mmvq.cu`
    - `ggml/src/ggml-cuda/vecdotq.cuh`
  - E: Wave64/MoE Fix
  - F: DPP Warp Utils
    - `ggml/src/ggml-cuda/common.cuh`
  - G: Build-System
    - `ggml/src/ggml-cuda/CMakeLists.txt`
  - H: Scripts & Tools

- **Welche Dateien geändert sind:**
  - `ggml/src/ggml-cuda/common.cuh`
  - `ggml/src/ggml-cuda/fattn*.cu*`
  - `ggml/src/ggml-cuda/mmq.cu`
  - `ggml/src/ggml-cuda/mmq.cuh`
  - `ggml/src/ggml-cuda/mmvq.cu`
  - `ggml/src/ggml-cuda/vecdotq.cuh`
  - `ggml/src/ggml-cuda/gfx906/` (neue Dateien)
  - `ggml/src/ggml-cuda/CMakeLists.txt`

- **Detaillierte Code-Änderungen** für jede Kategorie

**Nutze diese Datei als Referenz beim Mergen/Patchen!**

---

## 🎯 ARBEITSABLAUF

### Standard-Workflow: Rebase & Patch-Reparatur

Da dies ein Fork mit GFX906-Optimierungen ist, der regelmäßig auf upstream llama.cpp rebased wird:

```
1. Rebase auf upstream llama.cpp durchführen
   git fetch upstream
   git rebase upstream/master

2. BEI KONFLIKTEN: GFX906-Optimierungen wiederherstellen
   → Siehe Szenario B: "Konflikte nach Rebase lösen"

3.Commit mit aussagekräftiger Message
```

### Schritt 1: Kontext lesen (5 min)
```
1. Öffne plans/Optimierung.md
2. Verstehe: Was ist das Problem? Was ist die Lösung?
3. Öffne patches_komplett/cuda_ohne_gfx906.diff
4. Verstehe: Welche Kategorien sind relevant für DEINE Aufgabe?
```

### Schritt 2: Aufgabe verstehen (2 min)
```
1. Lies die konkrete Aufgabe (z.B. "Patch Kategorie C anwenden")
2. Finde die relevanten Kategorien in Diffs-Datei
3. Schau dir die Dateien an, die geändert werden
```

### Schritt 3:
- Commit mit aussagekräftiger Message
```

---

## 💡 TIPPS FÜR DIESE AUFGABEN

### ✅ DO's

- **Lese ALLE Kontextdateien** bevor du fragst
- **Teste nach jedem Patch** — Build + Benchmark
- **Committe nach erfolgreichen Tests** (nicht vorher!)
- **Nutze git branches** — nicht direkt in master arbeiten
- **Sei systematisch** — eine Kategorie nach der anderen


### ❌ DON'Ts

- **Blinde Patches anwenden** ohne zu verstehen was sie tun
- **Ganze Forks integrieren** ohne zu testen
- **Struktur-Annahmen machen** — prüfe ob Datei/Struct noch existiert
- **Build-Fehler ignorieren** — diagnostic immer sofort

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

### Szenario B: "Konflikte nach Rebase lösen"

**Situation:** Nach `git rebase upstream/master` gibt es Konflikte in Dateien mit GFX906-Optimierungen.

**Wichtig:** Die Original-Version von llama.cpp behalten und dann die GFX906-Optimierungen manuell wieder anwenden!

**Vorgehen:**

1. **Konfliktdatei identifizieren** (z.B. `ggml/src/ggml-cuda/common.cuh`)

2. **Für den Konflikt:**
   ```bash
   # Akzeptiere die ORIGINAL-Version von llama.cpp
   git checkout --ours ggml/src/ggml-cuda/common.cuh
   git add ggml/src/ggml-cuda/common.cuh
   ```

3. **GFX906-Optimierungen wieder anwenden:**
   ```
   a) Öffne `patches_komplett/cuda_ohne_gfx906.diff`
   b) Suche nach der betroffenen Datei (z.B. `common.cuh`)
   c) Vergleiche: Welche GFX906-Optimierungen fehlen jetzt?
   d) Wende die Optimierungen manuell an
   ```

4. **Beispiel aus heute (31.01.26):**
   - Konflikt in `common.cuh` bei `ggml_graph_node_properties`
   - Original llama.cpp hat `node_type` hinzugefügt + Reihenfolge geändert
   - Fehlerhafte Merge: Dupliziertes `int32_t flags`
   - Lösung: Erstes `flags` entfernen, korrekte Reihenfolge beibehalten

5. **Build testen:**
   ```bash
   cmake --build build 2>&1 | head -50
   ```

6. **Wenn Build erfolgreich:**
   ```bash
   git add <datei>
   git rebase --continue
   ```

**Typische Konflikt-Dateien:**
- `ggml/src/ggml-cuda/common.cuh` (Struktur-Änderungen, DPP Utils)
- `ggml/src/ggml-cuda/fattn*.cu*` (Flash-Attention Dispatch)
- `ggml/src/ggml-cuda/mmq.cu` / `mmq.cuh` (Quantization)
- `ggml/src/ggml-cuda/mmvq.cu` (Matrix-Vector Quantization)
- `ggml/src/ggml-cuda/vecdotq.cuh` (Vector Dot Quantization)
- `ggml/src/ggml-cuda/CMakeLists.txt` (Build-System)

---
