## Dev-Protokoll – Optimierung B (gfx906 Dispatch)

- **Kontext**: Kategorie B aus dem gfx906-Port – Einbindung der warp-kooperativen MMVQ-Kernel für MI50 (gfx906) in den Dispatch-Pfad.

- **Änderungen**:
  - [ggml/src/ggml-cuda/mmvq.cu](ggml/src/ggml-cuda/mmvq.cu#L1-L40): Guarded Includes für `gfx906/gfx906-mmvq-q*_*.cuh`, nur aktiv wenn `GGML_HIP_GFX906` gesetzt ist.
  - [ggml/src/ggml-cuda/mmvq.cu](ggml/src/ggml-cuda/mmvq.cu#L476-L560): Für `GGML_TYPE_Q4_0`, `Q4_1`, `Q8_0` wird bei `ncols_dst == 1`, `ncols_x <= 1024` und ohne Fusion auf die warp-kooperative `gfx906_launch_mul_mat_vec_*` Funktion verzweigt; sonst greift der bisherige generische Pfad.

- **Tests**:
  - `llama-server` mit `Qwen3-8B-DeepSeek-v3.2-Speciale-Distill.q8_0.gguf` gestartet (`GGML_HIP_DEBUG=1`, `--cont-batching 0`, `--cache-ram 0`).
  - Bei diesem Modell (`n_embd = 4096`) wurde der Warp-Coop-Pfad nicht getriggert, weil das Hard-Limit `ncols_x <= 1024` nicht erfüllt ist – im Debug-Log erscheinen nur die generischen Kernel.

- **Erkenntnis**:
  - Die gfx906 Warp-Coop-Kerne sind auf schmale Matrizen (≤1024 Spalten) zugeschnitten, also eher relevant für kleinere 1B/3B-Modelle oder MoE-Expertenblöcke mit geringer Dimension.
  - Für breite Matrizen (4k) liefert der Standard-MMVQ-Pfad weiterhin die bessere Performance; ein einfaches Anheben des Limits würde vermutlich zu schlechterer Auslastung führen.

- **Offene Punkte / Nächste Schritte**:
  1. Optional neues Benchmarking mit einem schmalen q8_0/Q4 Modell (≤1024 dims), um die Warp-Coop-Kerne in Aktion zu vermessen.
  2. Falls nötig, eigenständige gfx906-Kernel für breitere Matrizen entwickeln (größere Tiles, angepasste LDS-Strides) statt nur das Limit hochzusetzen.
