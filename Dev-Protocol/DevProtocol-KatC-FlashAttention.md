# Dev-Protocol Kategorie C  Flash Attention fr gfx906

## Kontext
- Basis: llama.cpp mainline `b7871`
- Ziel: Port der gfx906-spezifischen Flash-Attention Optimierungen (Kategorie C aus Optimierung.md)
- Zeitraum: Januar 2026

## Arbeitsschritte

### 1. Analyse der Fork-Diffs
- Dokument `plans/Optimierung.md` und `plans/Diffs-gfx906-mainline.md` gesichtet
- Kategorie C identifiziert: 
  - `ggml/src/ggml-cuda/fattn-common.cuh`
  - `ggml/src/ggml-cuda/fattn.cu`
  - neue gfx906-Kernel unter `ggml/src/ggml-cuda/gfx906/`
  - Build-System-Anpassungen (CMake)

### 2. Port der Common-Änderungen
- Datei `ggml/src/ggml-cuda/fattn-common.cuh`
  - AMD-spezifische Reduktionspfade integriert (wave32 DPP, Split-K Clamp)
  - HIP-Only Guards ergänzt
- Datei `ggml/src/ggml-cuda/common.cuh`
  - gfx906 DPP-Helfer inkludiert (`#include "gfx906/gfx906-common.cuh"`)

### 3. Flash-Attention Implementierung
- Datei `ggml/src/ggml-cuda/fattn.cu`
  - gfx906-spezifische Includes (`gfx906-fattn-q8.cuh`)
  - Tile-Dispatch erweitert: neue `GGML_CUDA_FATTN_TILE_*` Fälle fr Q8 und HIP
  - Env-Gates `GGML_HIP_FATTN_*` hinzugefgt
- Neue Dateien erstellt unter `ggml/src/ggml-cuda/gfx906/`:
  - `gfx906-common.cuh`, `gfx906-config.h`
  - Kernelfiles `gfx906-fattn-q8.cuh` + `gfx906-fattn-q8.cu`
  - Hilfsdateien fr MMQ/MMVQ/VEC-DOT (werden ebenfalls von Flash-Attn genutzt)
  - Template-Instanzen fr Q8 Tiles (`gfx906/template-instances/fattn-tile-q8-instance-*.cu`)

### 4. Build-System
- `ggml/src/ggml-cuda/CMakeLists.txt`
  - GLOBs erweitert, damit `gfx906/*.cuh|*.cu` sowie Q8-Template-Instanzen kompiliert werden
- `ggml/src/ggml-hip/CMakeLists.txt`
  - Analog CUDA: gfx906-Dateien eingebunden
  - Definition `GGML_HIP_GFX906` automatisch gesetzt, wenn Zielarchitektur `gfx906`

### 5. Laufzeit-Verifikation
- HIP-Build neu erzeugt (`cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx906`)
- `nm -D build/bin/libggml-hip.so | grep flash_attn_ext_tile_q8` verifiziert definierte Symbole
- `LD_DEBUG=libs ./build/bin/llama-server --help` kontrolliert, dass alle Backends korrekt geladen werden
- `llama-bench` mit `-fa 1` und gfx906-Modell durchlaufen lassen; Flash-Attention aktiv
- `llama-server` Testprompt (Qwen3-Coder Format) erfolgreich, ~59 tok/s

### 6. Warnungen / Follow-Up
- HIP-Compiler meldet `-Wpass-failed` bzgl. `amdgpu-waves-per-eu` (Occupancy-Ziel verfehlt). Funktional OK, optional `-Wno-pass-failed` setzen oder Attribute lockern.

## Ergebnis
- gfx906 Flash-Attention Pfad (Kategorie C) vollständig in mainline Port integriert.
- HIP- und CUDA-Builds ziehen automatisch die neuen Dateien.
- Laufzeit-Tests (bench + server) besttigen funktionierende Flash-Attention auf MI50.
