# Entwicklungsprotokoll: GFX906 Q5_K MMVQ (Warp‑Coop) + CMake Statusmeldungen

## Aufgabe: Q5_K_M GFX906‑Kernel für MMVQ + Build‑Statusausgabe

### Kontext
- **Ziel:** Beschleunigung häufiger Q5_K_M‑Modelle (kleine Matrizen) auf MI50/MI60 (gfx906) durch warp‑kooperative GEMV‑Kernels.
- **Kategorie:** A/B (GFX906 Kernel + Dispatch) und Build‑Transparenz
- **Betroffene Dateien:**
  - `ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q5_k.cuh`
  - `ggml/src/ggml-cuda/mmvq.cu`
  - `ggml/src/ggml-hip/CMakeLists.txt`

### Durchführung

#### 1) Neuer GFX906‑Kernel (Q5_K)
- **Datei:** `ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q5_k.cuh`
- **Inhalt:** Warp‑kooperativer GEMV‑Kernel (64 Threads → 2 Zeilen), angelehnt an Q4/Q8‑Kernels.
- **Kernidee:**
  - Half‑warp (32 Threads) pro Zeile.
  - Jeder Thread berechnet einen Teil‑Dot und reduziert per `warp_reduce_sum<32>`.
  - Nutzung von `vec_dot_q5_K_q8_1` (bestehende Dequant/VecDot‑Logik).
- **Korrektur:** Funktionsaufruf angepasst auf Signatur `vec_dot_q5_K_q8_1(vx, vy, kbx, iqs)`.

#### 2) Dispatch‑Erweiterung
- **Datei:** `ggml/src/ggml-cuda/mmvq.cu`
- **Änderungen:**
  - Include `gfx906-mmvq-q5_k.cuh` unter `GGML_HIP_GFX906`.
  - Dispatch für `GGML_TYPE_Q5_K` ergänzt:
    - Aktiv nur für **ncols_dst == 1**, **kein Fusion**, **ncols_x <= 1024**.
    - Fallback bleibt generischer MMVQ‑Kernel.

#### 3) Build‑Statusausgaben (HIP)
- **Datei:** `ggml/src/ggml-hip/CMakeLists.txt`
- **Ziel:** Sichtbarkeit der GFX906‑Quellen im CMake‑Output (wie im Fork).
- **Ausgaben:**
  - Liste gefundener `gfx906/*.cu`.
  - Liste gefundener `gfx906/template-instances/*.cu`.
  - Status‑Hinweis „GFX906 optimizations enabled (warp‑cooperative Q4/Q5/Q8 MMVQ)“.

### Ergebnis
- **Status:** ✅ Implementiert, buildfähig nach Fix.
- **Nächste Schritte:**
  1. **Rebuild** (`cmake ..` + `make -j`).
  2. **Benchmark Q5_K_M** (kleine Matrizen / MoE‑ähnliche Fälle), Vergleich zu vorher.

### Anmerkungen
- Q5_K‑Optimierung wirkt primär auf **MMVQ‑Pfad** (kleine Matrizen). Für große Matrizen wäre ein späterer **MMQ‑Pfad**‑Tuning nötig.
- CMake‑Status ist rein informativ und ändert keine Build‑Logik.
