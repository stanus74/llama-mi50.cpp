# Entwicklungsprotokoll: Kategorie D - Q8/MXFP4 Pipeline Optimierungen

## Aufgabe: Integration der gfx906 Pipeline-Optimierungen für Q8_0 und MXFP4

### Kontext
- **Ziel:** Verbesserung der Inferenz-Performance auf MI50/MI60 (gfx906) durch Software-Pipelining, Vectorized Loads und reduzierten LDS-Bank-Conflicts.
- **Kategorie:** D (Quantization Pipeline)
- **Quelle:** `plans/D-q8-mxfp4-pipeline.patch`
- **Betroffene Dateien:**
  - `ggml/src/ggml-cuda/mmq.cuh`
  - `ggml/src/ggml-cuda/vecdotq.cuh`

### Durchführung

#### 1. Analyse Patch D
Der Patch enthielt umfangreiche Änderungen an den MMQ-Kerneln (Matrix Multiplication Quantized) und Vector-Dot-Product Funktionen. Hauptkomponenten:
- Integration neuer Header (`gfx906-mmq.cuh`, `gfx906-vecdotq.cuh`).
- Umstellung von `load_tiles` Funktionen auf asynchrone/gepipelinte Versionen für GFX906.
- Einführung von Prefetching (`gfx906_prefetch_y_tile_v4`) in der Hauptschleife `mul_mat_q_process_tile`.

#### 2. Manuelle Integration (`mmq.cuh`)
Da der Patch nicht direkt anwendbar war (Kontext-Unterschiede), wurden die Änderungen blockweise manuell eingepflegt:
- **Includes & Macros:** GFX906-Header und `MMQ_ITER_K` Definitionen hinzugefügt.
- **LDS Stride:** `MMQ_TILE_Y_K_LDS` eingeführt, um Optimierungen bei Bank Conflicts zu ermöglichen (bzw. Padding-Experimente zu dokumentieren).
- **`load_tiles_q8_0`:** Implementierung von `GFX906_LOAD_TILES_Q8_0_ASYNC` für HIP/gfx906. Korrektur des `need_check` Pfads zur Vermeidung serialisierter LDS-Writes durch Out-of-Bounds Threads.
- **`load_tiles_mxfp4`:** Implementierung einer Software-Pipeline (2-Phasen: Load -> Dequant/Store) zur Maximierung der Memory-Level-Parallelism.
- **`mul_mat_q_process_tile`:** Einfügen des Prefetching-Logik-Blocks (`gfx906_prefetch_y_tile_v4` und `gfx906_prefetch_consume`) zwischen Barrieren und Berechnungen.
- **`vec_dot_*` Kernels:** Aktualisierung der Schleifen auf `MMQ_TILE_Y_K_LDS` und Verwendung von `gfx906_load_q4_*_vectorized` Helpern.

#### 3. Manuelle Integration (`vecdotq.cuh`)
- **Bitwise Permutation:** `get_int_from_table_16` wurde für HIP optimiert. Anstatt `__builtin_amdgcn_perm` (was teilweise Instruktionen nutzt, die CUDA `__byte_perm` nicht 1:1 entsprechen) wurde eine explizite Bitwise-Implementierung mit Maskierung gewählt, die für HIP korrekt kompiliert.
- **MXFP4 Unterstützung:** `get_int_from_mxfp4_table` leitet nun auf die optimierte GFX906-Variante um.
- **Kernel-Optimierungen:**
  - `vec_dot_mxfp4_q8_1`: nutzt `GFX906_VEC_DOT_MXFP4_Q8_1`.
  - `vec_dot_q8_0_q8_1`: nutzt `gfx906_get_int_b2_fast` (optimierte 16-Bit Loads).

### Ergebnis
- **Status:** ✅ Code-Integrationsphase abgeschlossen.
- **Nächste Schritte:** 
  1. Kompilierung prüfen (`make`).
  2. Benchmarks durchführen (`llama-bench`), um Performance-Gewinn zu verifizieren (Erwartung: Stabilere Eval-Zeiten bei großen Prompts).

### Anmerkungen / Herausforderungen
- Der Diff für `vec_dot_q4_0_q8_1_dp4a` und `vec_dot_q4_1_q8_1_dp4a` musste sorgfältig angepasst werden, um sicherzustellen, dass sowohl der GFX906-Pfad als auch der generische Fallback-Pfad korrekt sind (besonders bezüglich `MMQ_TILE_Y_K_LDS`).
- `get_int_from_table_16`: Hier wurde die Logik angepasst, um Kompatibilitätsprobleme mit `__byte_perm` vs AMD intrinsics zu umschiffen.
