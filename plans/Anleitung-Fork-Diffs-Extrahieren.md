# 📦 Anleitung: gfx906-Änderungen aus Fork extrahieren

## Schnell-Übersicht

Wenn eine neue Fork-Release von **iacopPBK** verfügbar ist, funktioniert die Extraktion so:

```bash
# 1. Fork neu clonen (oder aktualisieren)
git clone https://github.com/iacopPBK/llama.cpp-gfx906.git fork_neu
cd fork_neu

# 2. Mainline-Version als Referenz abrufen
git fetch https://github.com/ggerganov/llama.cpp.git b7924
git tag mainline-reference FETCH_HEAD

# 3. Alle Unterschiede extrahieren
git diff mainline-reference..HEAD -- ggml/src/ggml-cuda/ > cuda_gfx906_all.diff

# 4. Optional: Nach Kategorien trennen
git diff mainline-reference..HEAD -- ggml/src/ggml-cuda/gfx906/ > gfx906_kernels.diff
git diff mainline-reference..HEAD -- ggml/src/ggml-cuda/mmvq.cu > gfx906_mmvq_dispatch.diff
git diff mainline-reference..HEAD -- ggml/src/ggml-cuda/fattn*.cu* > gfx906_flashattn.diff
```

---

## 📋 Detaillierte Schritte

### Schritt 1: Fork klonen

```bash
# Neuen Fork klonen oder aktualizieren
git clone https://github.com/iacopPBK/llama.cpp-gfx906.git llama-gfx906-neu
cd llama-gfx906-neu
```

### Schritt 2: Mainline-Basis bestimmen

```bash
# Fork-Status prüfen (schau auf "X commits ahead/behind")
git log --oneline -1

# Die Nummer auf GitHub anschauen: iacopPBK/llama.cpp-gfx906
# Z.B.: "This branch is 9 commits ahead of and 5 commits behind ggml-org/llama.cpp:master"
```

**Wichtig:** Der Fork basiert meist auf einer älteren Mainline-Version (z.B. b7924, b7945, b7970).

### Schritt 3: Mainline als Referenz abrufen

```bash
# Die Mainline-Version fetchen (z.B. b7924 - oder was der Fork verwendet)
git fetch https://github.com/ggerganov/llama.cpp.git b7924
git tag mainline-ref FETCH_HEAD

# Oder: Wenn du die Nummer nicht kennst, schau in den Changelog des Forks
```

### Schritt 4: Unterschiede anschauen

```bash
# Schnell-Übersicht: Welche CUDA-Dateien wurden geändert?
git diff mainline-ref..HEAD --stat -- ggml/src/ggml-cuda/

# Das sollte zeigen:
#  ggml/src/ggml-cuda/gfx906/  (viele neue Dateien)
#  ggml/src/ggml-cuda/mmvq.cu  (Änderungen)
#  ggml/src/ggml-cuda/fattn*.cu* (Änderungen)
#  etc.
```

### Schritt 5: Diffs extrahieren

**Option A: Alles auf einmal**
```bash
git diff mainline-ref..HEAD -- ggml/src/ggml-cuda/ > cuda_gfx906_complete.diff
```

**Option B: Nach Kategorien** (wie in `plans/Optimierung.md`)
```bash
# Kategorie A: Neue gfx906-Kernel
git diff mainline-ref..HEAD -- ggml/src/ggml-cuda/gfx906/ > A-gfx906-kernels.diff

# Kategorie B: Dispatch/Registrierung
git diff mainline-ref..HEAD -- ggml/src/ggml-cuda/mmvq.cu > B-gfx906-dispatch.diff

# Kategorie C: Flash-Attention
git diff mainline-ref..HEAD -- ggml/src/ggml-cuda/fattn*.cu* > C-gfx906-flashattn.diff

# Kategorie D: Quantization
git diff mainline-ref..HEAD -- ggml/src/ggml-cuda/mmq.cu > D-gfx906-quantization.diff

# Extras: CMake, Common
git diff mainline-ref..HEAD -- ggml/src/ggml-cuda/CMakeLists.txt > cmake-changes.diff
git diff mainline-ref..HEAD -- ggml/src/ggml-cuda/common.cuh > common-changes.diff
```

**Option C: Patches (Commit-basiert)**
```bash
git format-patch mainline-ref -o patches_fork/
# Generiert: 0001-*.patch, 0002-*.patch, etc.
```

---

## 🛠️ Häufige Probleme

### Problem: Tag/Commit nicht gefunden
```bash
# Fehler: "Konnte Remote-Referenz b7924 nicht finden"

# Lösung: Mit FETCH_HEAD arbeiten
git fetch https://github.com/ggerganov/llama.cpp.git b7924
git diff FETCH_HEAD..HEAD -- ggml/src/ggml-cuda/
```

### Problem: Fork ist schon auf neuerer Mainline Version
```bash
# Der Fork wurde inzwischen mit neuerer Mainline gemerged
# Z.B. b7924 → b7945 → b7970

# Lösung: 
# 1. Rausfinden auf welche Version der Fork basiert
git log --oneline | grep "Merge branch"  # Schau die Merges

# 2. Oder: Einfach gegen die README-Angabe gehen
# (die meisten Forks dokumentieren die Basis-Version)

# 3. Alternativ: Schaue auf GitHub bei der Release
# "Based on llama.cpp build XXXX"
```

---

## 📊 Validierung

Nach dem Extrahieren solltest du prüfen:

```bash
# 1. Diff-Größe prüfen (sollte nicht winzig sein)
wc -l cuda_gfx906_complete.diff
# Erwartet: mehrere 1000 Zeilen

# 2. Inhalt prüfen (gibt es gfx906-Dateien?)
grep "gfx906" cuda_gfx906_complete.diff | head -5

# 3. Hast du alles was in Optimierung.md dokumentiert ist?
grep "mmvq-q4_0\|mmvq-q4_1\|mmvq-q8_0\|fattn-q8" cuda_gfx906_complete.diff
```

---

## 🔄 Workflow für neue Fork-Releases

1. **Auf GitHub prüfen:** https://github.com/iacopPBK/llama.cpp-gfx906/releases
2. **"Based on llama.cpp build XXXX" Nummer notieren** (z.B. b7924)
3. **Fork klonen** und die Schritte 2-5 oben durchführen
4. **Diffs speichern** in deinem `patches_komplett/` Verzeichnis
5. **Testen:** Mit den neuen Patches gegen deine Mainline-Version bauen
6. **Dokumentieren:** Notiz in `Optimierung.md` machen (neue Version, Änderungen)

---

## 💡 Tipps

- **Immer Git verwenden**, nie manuell Dateien kopieren
- **Mainline-Version dokumentieren** (b7924, b7945, etc.) — sie ist wichtig für zukünftige Merges
- **Diff-Dateien versionieren** in `patches_komplett/` mit Dateinamen wie `cuda_vs_b7924.diff`
- **Nach großen Änderungen** einen neuen Branch erstellen bevor du Patches anwendest

---

## 🎯 Beispiel: Neue Release 2603 (Zukunft)

Angenommen es gibt eine neue Release 2603 basierend auf **b7950**:

```bash
# 1. Fork neu klonen
git clone https://github.com/iacopPBK/llama.cpp-gfx906.git fork_2603
cd fork_2603

# 2. Mainline b7950 abrufen
git fetch https://github.com/ggerganov/llama.cpp.git b7950
git tag mainline-b7950 FETCH_HEAD

# 3. Diffs extrahieren
git diff mainline-b7950..HEAD -- ggml/src/ggml-cuda/ > /path/to/patches_komplett/cuda_vs_b7950.diff
git diff mainline-b7950..HEAD -- ggml/src/ggml-cuda/gfx906/ > /path/to/patches_komplett/gfx906_kernels_vs_b7950.diff

# 4. Speichern und dokumentieren
# → Neue Einträge in Optimierung.md hinzufügen
```

Fertig! Die neuen Diffs kannst du dann wie gehabt anwenden.
