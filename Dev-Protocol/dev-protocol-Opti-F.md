# Dev Protocol – Opti-F

## Hintergrund
- Ziel: Verbesserte HIP/GFX906-Pfade und konsistentes CUDA-Graph-Handling.
- Kontext: Mainline hat `ggml_cuda_graph_node_properties` → `ggml_graph_node_properties` refactor durchgeführt.

## Änderungen
1. **CUDA Common Anpassungen**
   - `ggml/src/ggml-cuda/common.cuh`
     - Neuer helper `ggml_cuda_shfl_xor_sync` für einheitliche `shfl_xor`-Verwendung.
     - HIP-Spezialpfade nutzen DPP/HIP-Shuffles (über neuen Header `gfx906/gfx906-common.cuh`).
     - Warp-Reduktionsfunktionen (`warp_reduce_sum/max`, `warp_reduce_all/any`) verwenden Helper.
     - Fast-Path für `ggml_cuda_e8m0_to_fp32` bei `__gfx906__` (direktes Bit-Cast).
     - CUDA-Graph-Struktur um Felder `cuda_graphs_enabled`, `disable_due_to_failed_graph_capture`, `ggml_graph_properties` erweitert.

2. **Neuer Header für GFX906**
   - `ggml/src/ggml-cuda/gfx906/gfx906-common.cuh`
     - Stellt DPP-Hilfsfunktionen (`hip_dpp_xor*`) bereit.
     - Enthält generische `warp_reduce_amd_f32`-Implementierung mit `AddOp`/`MaxOp`.

3. **CUDA Backend Aktualisierung**
   - `ggml/src/ggml-cuda/ggml-cuda.cu`
     - `ggml_cuda_graph_node_set_properties` und `ggml_cuda_graph_node_properties_match` jetzt auf `ggml_graph_node_properties` portiert.
     - `graph->props` → `graph->ggml_graph_properties` umgezogen, Feldzugriffe aktualisiert.

## Status & Tests
- HIP-Build mit gfx906 erneut gestartet; Fehler durch fehlende Typ-Anpassung behoben.
- Keine automatischen Tests ausgeführt – Fokus lag auf Build-Fix.

## Nächste Schritte
- Sicherstellen, dass `cuda_graphs_enabled` an passenden Stellen aktiviert wird.
- Weitere HIP-spezifische Performance-Messungen durchführen, um Nutzen der neuen DPP-Fallbacks zu verifizieren.
