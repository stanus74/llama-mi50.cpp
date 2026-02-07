## Dev-Protokoll – Optimierungen E & G

- **Kontext**: Kategorie E (Wave64/MoE Fix) plus Kategorie G (gfx906 Build-System) laut Projektplan; Ziel ist ein stabiler MoE-Pfad auf MI50 sowie reproduzierbare HIP/CUDA-Build-Artefakte.

- **Änderungen**:
  - [ggml/src/ggml-cuda/mmid.cu](ggml/src/ggml-cuda/mmid.cu#L138-L170): HIP-Wave64-Fallback für `n_expert_used >= warp_size/2`, so dass große MoE-Batches auf MI50 automatisch die sichere generische Variante nutzen.
  - [ggml/src/ggml-cuda/mmq.cu](ggml/src/ggml-cuda/mmq.cu#L6-L368): `s13`-Stride korrigiert, RDNA3/RDNA4-MMQ-Heuristiken verfeinert und dabei versehentlich zerstörte Logik (Dispatch-Switch, `ggml_cuda_op_mul_mat_q`, `ggml_cuda_should_use_mmq`) vollständig restauriert.
  - [ggml/src/ggml-cuda/CMakeLists.txt](ggml/src/ggml-cuda/CMakeLists.txt#L47-L58): Sobald CUDA ≥ 12.8 erkannt wird, hängen wir jetzt beide Blackwell-Architekturen (`120a-real`, `121a-real`) gemeinsam an, damit ältere CMake-Versionen deren Validierung bestehen.

- **Tests**: Noch kein neuer Build – nächster Schritt ist `cmake -B build -DGGML_HIP=ON && cmake --build build -j` auf gfx906.

- **Offene Punkte**:
  1. HIP-Build durchlaufen lassen und prüfen, ob der Wave64-Fallback einen sauberen Launch liefert.
  2. MoE-Benchmark (hohe `n_experts`) mit MI50 fahren, um den Fallback-Pfad zu triggern und die Performance zu vermessen.
  3. Nach Integration weiterer gfx906-Kategorien (A/B/D) erneut bauen und Benchmarks aktualisieren.
