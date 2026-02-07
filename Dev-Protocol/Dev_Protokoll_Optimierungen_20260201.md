# Dev‑Protokoll – Optimierungen (2026‑02‑01)

## Zusammenfassung
- Optimierung 1, 2 und 4 haben Performance verbessert.
- Optimierung 3 (LDS‑Caching) wurde verworfen.

## Details

### Optimierung 1 – VDR‑Tuning (Q5_K)
- Änderung: `VDR_Q5_K_Q8_1_MMVQ` von 2 → 4
- Dateien: ggml/src/ggml-cuda/vecdotq.cuh
- Ergebnis: Performance verbessert

### Optimierung 2 – 32‑Thread Kernel (1 Row/Block)
- Änderung: zusätzlicher Kernel + Dispatch
- Heuristik: `nrows_x < 32`
- Dateien:
  - ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q5_k.cuh
  - ggml/src/ggml-cuda/mmvq.cu
- Ergebnis: Performance verbessert

### Optimierung 3 – LDS‑Caching (Q8‑Tiles)
- Änderung: Shared‑Memory Cache pro `ib`
- Dateien: ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q5_k.cuh
- Ergebnis: Performance schlechter → rückgängig gemacht

### Optimierung 4 – Loop‑Unroll (ib += 2)
- Änderung: Unroll im Q5_K‑Kernel
- Dateien: ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q5_k.cuh
- Ergebnis: Performance verbessert

## Status
- 1: ✅
- 2: ✅
- 3: ❌ (reverted)
- 4: ✅
