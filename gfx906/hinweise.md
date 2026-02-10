# GFX906 Hinweise (MI50)

Kurznotizen fuer MI50/gfx906 Optimierungen in diesem Repo. Fokus: DPP/SDWA, VOP2/VOP3/VOP3P und praktische Checks.

## Instruktions-Syntax (gfx906)
- VOP2/VOP3/VOP3P bieten DPP/SDWA Varianten fuer Shuffle, Dot- und FMA Pfade.
- DPP/SDWA haben Legalitaets- und Permutations-Limits; ungueltige Kombinationen koennen still scheitern oder langsame Fallbacks ausloesen.
- VOP3P Dot-Operationen sind relevant fuer quantisierte Dot/VecDot Pfade.

## Praktische Validierung
- DPP/SDWA Pfade pro Kernelgruppe pruefen (Legalitaet, Permutationen, Wave64 Verhalten).
- LDS-Layout/Stride testen (z.B. 32/33/40/41) um Bankkonflikte zu minimieren.
- Alignment fuer vectorized loads (int4/float4) in quantisierten Pfaden pruefen.

## Relevante Repo-Pfade
- Aufgabenliste: [tasks.md](../tasks.md)
- GFX906 Basis-Header: [ggml/src/ggml-cuda/gfx906/gfx906-common.cuh](../ggml/src/ggml-cuda/gfx906/gfx906-common.cuh)
- GFX906 Konfiguration: [ggml/src/ggml-cuda/gfx906/gfx906-config.h](../ggml/src/ggml-cuda/gfx906/gfx906-config.h)
- Flash-Attention: [ggml/src/ggml-cuda/fattn.cu](../ggml/src/ggml-cuda/fattn.cu)
- Quantisierung/VecDot: [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh), [ggml/src/ggml-cuda/vecdotq.cuh](../ggml/src/ggml-cuda/vecdotq.cuh)

## Profiling-Hinweise
- Pro Kategorie messen: Occupancy, VGPR, LDS, VMEM/SMEM, Wavefronts/SE.
- Microbenchmarks fuer MMQ/MMVQ/FATTN/VecDot separat laufen lassen.

## Wave64 Checkliste fuer FATTN
- Alle warp_reduce_* Stellen pruefen: Default ist 32, fuer Wave64 ggf. bewusst ok oder explizit 64 setzen.
- Teilaktive Lanes testen (maskierte Threads) und KQ_max/KQ_sum korrekt verifizieren.
- FATTN Q8 Tile Kernel: Head-Size Filter via GFX906_Q8_SUPPORTS_HEAD_DIM konsistent halten.

## !!! Roadmap aus gfx906/readme bezieht sich auf  iacopPBK/llama.cpp-gfx906 fork 

(A-F)
- F: DPP-basierte Warp-Reduktionen sind bereits stark optimiert; nur validieren und regressionsfrei halten.
- C: Flash-Attention ist GCN-getuned; Aenderungen nur nach Profiling.
- D: MXFP4 Dequantization ist vorhanden, GPU-Quantization pruefen/erganzen.
- A/B: gfx906 Kernels und Dispatch eher validieren/profilen als neu erfinden.
- E: Wave64/Korrektheitstests besonders bei teilaktiven Lanes und MoE achten.
