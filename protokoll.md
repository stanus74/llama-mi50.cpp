# Protokoll

Datum: 2026-02-07

## Ziel
Umsetzung der Kategorie F (GPU Utility) und Kategorie C (Flash-Attention) gemaess diff-b7924.

## Kategorie F - GPU Utility / Basis-Optimierung
- DPP/Shuffle-Optimierungen und HIP-spezifische Pfade in [ggml/src/ggml-cuda/common.cuh](ggml/src/ggml-cuda/common.cuh).
- Neue GFX906 Konfiguration in [ggml/src/ggml-cuda/gfx906/gfx906-config.h](ggml/src/ggml-cuda/gfx906/gfx906-config.h).
- Neue GFX906 DPP/Shuffle Hilfsfunktionen in [ggml/src/ggml-cuda/gfx906/gfx906-common.cuh](ggml/src/ggml-cuda/gfx906/gfx906-common.cuh).

## Kategorie C - Flash-Attention (GCN/CDNA Tuning)
- Kommentar-Update und DPP Shuffle in [ggml/src/ggml-cuda/fattn-common.cuh](ggml/src/ggml-cuda/fattn-common.cuh).
- Split-K Heuristik fuer AMD PP vs TG in [ggml/src/ggml-cuda/fattn-common.cuh](ggml/src/ggml-cuda/fattn-common.cuh).
- GFX906 Q8 Kernel-Auswahl und Dispatch in [ggml/src/ggml-cuda/fattn.cu](ggml/src/ggml-cuda/fattn.cu).
- Nutzung der GFX906 Q8 Kernels in [ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cu](ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cu) und [ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cuh](ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cuh).

## Hinweise
- Externe Aenderungen an gfx906-config.h und gfx906-common.cuh wurden beibehalten.
- Weitere Kategorien laut tasks.md sind noch offen.
