# Tasks

## Priorisierte Reihenfolge
- Kategorie F (GPU Utility / Basis-Optimierung)
- Kategorie C (Flash-Attention)
- Kategorie E (Wave64 / MoE Korrektheits-Fix)
- Kategorie G (Build-System / HIP Integration)
- Kategorie A (MI50 / GFX906 Kernels)
- Kategorie B (Kernel-Dispatch / Registrierung)
- Kategorie D (Quantization Pipeline)
- Kategorie H (Tooling / Scripts)

## Kategorie A - MI50 / GFX906 Kernels
- Identifiziere alle neuen Dateien unter ggml/src/ggml-cuda/gfx906/ (attention, fused, matmul, quantize).
- Verifiziere, dass die Dateien nur gfx906-spezifische Implementierungen enthalten.
- Pruefe Abhaengigkeiten zu allgemeinen CUDA/HIP Headern und passe Includes falls noetig an.
- Dokumentiere Kernel-Gruppen nach Zweck (MMQ, MMVQ, FATTN, VEC/DOT, Fused Ops).
- Validierung: DPP/SDWA Legalitaet je Kernelgruppe pruefen (keine ungueltigen Permutationen).
- Validierung: LDS-Layout/Stride Matrix testen (z.B. 32/33/40/41), Bankkonflikte minimieren.
- Validierung: Alignment fuer vectorized Loads (int4/float4) dokumentieren und pruefen.
- Aenderungen umsetzen (Quelle: diff-b7924/all_gfx906_changes.diff, diff-b7924/cuda_gfx906_changes.diff):
- GFX906 Q8 Flash-Attention Kernels hinzufuegen: [ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cu](ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cu), [ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cuh](ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cuh).
- GFX906 Attention Tile-Instanzen anlegen: [ggml/src/ggml-cuda/gfx906/attention/instances/](ggml/src/ggml-cuda/gfx906/attention/instances/).
- GFX906 Attention RoPE Helfer anlegen: [ggml/src/ggml-cuda/gfx906/attention/rope.cuh](ggml/src/ggml-cuda/gfx906/attention/rope.cuh).
- Fused Ops fuer Q8 Pipeline anlegen: [ggml/src/ggml-cuda/gfx906/fused/](ggml/src/ggml-cuda/gfx906/fused/).
- Matmul Kernels und Prefetch anlegen: [ggml/src/ggml-cuda/gfx906/matmul/](ggml/src/ggml-cuda/gfx906/matmul/).
- Quantize Helfer und Q8 Cache anlegen: [ggml/src/ggml-cuda/gfx906/quantize/](ggml/src/ggml-cuda/gfx906/quantize/).
- ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cu: Q8 Flash-Attention Kernel fuer gfx906.
- ggml/src/ggml-cuda/gfx906/attention/fattn-q8.cuh: Q8 Flash-Attention Header fuer gfx906.
- ggml/src/ggml-cuda/gfx906/attention/instances/: Tile-Instanzen fuer Q8 Flash-Attention.
- ggml/src/ggml-cuda/gfx906/attention/rope.cuh: GFX906 RoPE Hilfsfunktionen fuer Attention.
- ggml/src/ggml-cuda/gfx906/fused/: Fused Ops (gather, norm, graph fusion, prequantized mmq).
- ggml/src/ggml-cuda/gfx906/matmul/: MMQ/MMVQ/SGEMM Kernels und Prefetch.
- ggml/src/ggml-cuda/gfx906/quantize/: Quantize und VecDot Hilfsheader, Q8 Cache.

## Kategorie B - Kernel-Dispatch / Registrierung
- Pruefe die Dispatch-Logik in ggml/src/ggml-cuda/mmvq.cu auf gfx906-Pfade.
- Stelle sicher, dass Fallback-Pfade fuer Nicht-gfx906 korrekt bleiben.
- Verifiziere Guard-Makros fuer HIP und gfx906.
- Validierung: Guard-Makros gegen ROCm/HIP Versionen pruefen (keine stillen Fallbacks).
- Profiling-Gate: Pfadwahl protokollieren (Log/Trace), damit gfx906-Pfad sicher aktiv ist.
- Aenderungen umsetzen (Quelle: diff-b7924/all_gfx906_changes.diff, diff-b7924/cuda_gfx906_changes.diff):
- GFX906 Dispatch und Kernel-Auswahl in [ggml/src/ggml-cuda/mmvq.cu](ggml/src/ggml-cuda/mmvq.cu) integrieren.
- Backend-Integration fuer GFX906 in [ggml/src/ggml-cuda/ggml-cuda.cu](ggml/src/ggml-cuda/ggml-cuda.cu) nachvollziehen.
- ggml/src/ggml-cuda/mmvq.cu: Dispatch fuer MMVQ auf gfx906.
- ggml/src/ggml-cuda/ggml-cuda.cu: Backend-Integration und Pfadwahl.

## Kategorie C - Flash-Attention (GCN/CDNA Tuning)
- Status: erledigt.
- Vergleiche Aenderungen in ggml/src/ggml-cuda/fattn-common.cuh.
- Vergleiche Aenderungen in ggml/src/ggml-cuda/fattn.cu.
- Ordne GFX906-Q8 Attention-Kernel unter ggml/src/ggml-cuda/gfx906/attention ein.
- Notiere Aenderungen an Tile-Groessen und Split-K Entscheidungen.
- Validierung: DPP/Shuffle Pfade mit Wave64 Divergenz testen (EXEC Mask Szenarien).
- Profiling-Gate: Tile-Varianten vs. LDS/VMEM/Occupancy messen und dokumentieren.
- Aenderungen umsetzen (Quelle: diff-b7924/all_gfx906_changes.diff, diff-b7924/cuda_gfx906_changes.diff):
- Split-K Logik fuer PP vs TG anpassen in [ggml/src/ggml-cuda/fattn.cu](ggml/src/ggml-cuda/fattn.cu).
- Shuffles auf DPP/Unified Shuffle umstellen in [ggml/src/ggml-cuda/fattn-common.cuh](ggml/src/ggml-cuda/fattn-common.cuh).
- GFX906 Q8 Attention Pfad einhaengen in [ggml/src/ggml-cuda/fattn.cu](ggml/src/ggml-cuda/fattn.cu) und GFX906 Kernels unter [ggml/src/ggml-cuda/gfx906/attention/](ggml/src/ggml-cuda/gfx906/attention/).
- ggml/src/ggml-cuda/fattn-common.cuh: Flash-Attention Konfiguration und Hilfsroutinen.
- ggml/src/ggml-cuda/fattn.cu: Kernel-Auswahl und Split-K Logik.
- ggml/src/ggml-cuda/gfx906/attention/: GFX906 Q8 Attention Kernels und Instanzen.

## Kategorie D - Quantization Pipeline (Q8 / MXFP4 / VecDot)
- Pruefe Aenderungen in ggml/src/ggml-cuda/mmq.cuh.
- Pruefe Aenderungen in ggml/src/ggml-cuda/vecdotq.cuh.
- Ordne neue Quantize-Header unter ggml/src/ggml-cuda/gfx906/quantize ein.
- Notiere Optimierungen (Software-Pipelining, Vektorisierung, LDS-Strategien).
- Validierung: Alignment und Vector-Load Pfade (int4/float4) pro Quant-Typ pruefen.
- Validierung: LDS-Layout/Stride Matrix testen (Bankkonflikte bei Q8/MXFP4).
- Profiling-Gate: Prefetch-Tiefe vs. VGPR-Druck/Occupancy evaluieren.
- Microbenchmarks: VecDot/MMQ/MMVQ separat messen, um Regressions frueh zu erkennen.
- Aenderungen umsetzen (Quelle: diff-b7924/all_gfx906_changes.diff, diff-b7924/cuda_gfx906_changes.diff):
- Q8 Pipeline Anpassungen und Prefetch in [ggml/src/ggml-cuda/mmq.cuh](ggml/src/ggml-cuda/mmq.cuh).
- VecDot und quantisierte Dot Pfade in [ggml/src/ggml-cuda/vecdotq.cuh](ggml/src/ggml-cuda/vecdotq.cuh).
- GFX906 Quantize Helfer einbinden unter [ggml/src/ggml-cuda/gfx906/quantize/](ggml/src/ggml-cuda/gfx906/quantize/).
- ggml/src/ggml-cuda/mmq.cuh: Quantization Pipeline fuer MMQ und Load-Pfade.
- ggml/src/ggml-cuda/vecdotq.cuh: VecDot und quantisierte Dot-Optimierungen.
- ggml/src/ggml-cuda/gfx906/quantize/: gfx906 Quantize Helfer und Q8 Cache.

## Kategorie E - Wave64 / MoE Korrektheits-Fix
- Pruefe Fixes in ggml/src/ggml-cuda/mmq.cu.
- Pruefe Fixes in ggml/src/ggml-cuda/mmid.cu.
- Verifiziere Bedingungen fuer Fallback-Pfade bei grossen Expert-Zahlen.
- Notiere Auswirkungen auf Stabilitaet fuer MoE Modelle.
- Validierung: Divergenz/EXEC Mask Tests bei grossen Expert-Zahlen definieren.
- Validierung: Sub-Warp Shuffle Korrektheit in Wave64 mit teilaktiven Lanes pruefen.
- Aenderungen umsetzen (Quelle: diff-b7924/all_gfx906_changes.diff, diff-b7924/cuda_gfx906_changes.diff):
- Sub-Warp Shuffle Fix und Fallback-Regeln in [ggml/src/ggml-cuda/mmq.cu](ggml/src/ggml-cuda/mmq.cu).
- MoE Stabilitaetsfixes in [ggml/src/ggml-cuda/mmid.cu](ggml/src/ggml-cuda/mmid.cu).
- ggml/src/ggml-cuda/mmq.cu: Wave64 Shuffle Fix und MoE Pfade.
- ggml/src/ggml-cuda/mmid.cu: Korrektheitsfixes fuer grosse Expert-Zahlen.

## Kategorie F - GPU Utility / Basis-Optimierung
- Status: erledigt.
- Pruefe DPP-basierte Warp-Reduktionen und Shuffle-Dispatch in ggml/src/ggml-cuda/common.cuh.
- Pruefe gfx906 Hilfsheader in ggml/src/ggml-cuda/gfx906/gfx906-common.cuh und ggml/src/ggml-cuda/gfx906/gfx906-config.h.
- ggml/src/ggml-cuda/common.cuh: Warp-Reduktionen und Shuffle Dispatch (DPP Pfad).
- ggml/src/ggml-cuda/gfx906/gfx906-common.cuh: DPP Ops und Warp Utils fuer gfx906.
- ggml/src/ggml-cuda/gfx906/gfx906-config.h: Schalter und Konstanten fuer gfx906.
- Validierung: DPP/SDWA Legalitaet und Permutation-Limits dokumentieren.
- Aenderungen umsetzen (Quelle: diff-b7924/all_gfx906_changes.diff, diff-b7924/cuda_gfx906_changes.diff):
- Unified Shuffle XOR und DPP-Reduktionen in [ggml/src/ggml-cuda/common.cuh](ggml/src/ggml-cuda/common.cuh).
- GFX906 DPP Utils und Konfigurationen anlegen in [ggml/src/ggml-cuda/gfx906/gfx906-common.cuh](ggml/src/ggml-cuda/gfx906/gfx906-common.cuh) und [ggml/src/ggml-cuda/gfx906/gfx906-config.h](ggml/src/ggml-cuda/gfx906/gfx906-config.h).

## Kategorie G - Build-System / HIP Integration
- Status: erledigt.
- Pruefe Build-Integration fuer gfx906-Quellen in ggml/src/ggml-cuda/CMakeLists.txt.
- ggml/src/ggml-cuda/CMakeLists.txt: GFX906 Quellen in Build einbinden.
- Aenderung merken: gfx906 Globs hinzufuegen (gfx906/*.cuh, gfx906/*.cu, gfx906/fused/*.cu, gfx906/attention/*.cu, gfx906/attention/instances/*.cu).
- Diff-Quelle: diff-b7924/all_gfx906_changes.diff, diff-b7924/cuda_gfx906_changes.diff.

## Kategorie H - Tooling / Scripts
- Pruefe MI50 Build- und Bench-Skripte und aktualisiere Pfade/Flags bei Bedarf.
- SCRIPT_compile_MI50.sh: ROCm Build Flags und Build-Pipeline.
- SCRIPT_launch_server_MI50.sh: Server-Start mit MI50 Defaults.
- SCRIPT_llama_bench.sh: Benchmark-Defaults fuer MI50.
- SCRIPT_llama_bench2.sh: Alternative Benchmark-Konfiguration.
- SCRIPT_overclock_upp_MI50.sh: Overclocking Workflow (optional).
- scripts/bench-models.sh: Sammel-Benchmarks und Model-Loop.

## Querschnitt - Validierung und Profiling
- Profiling-Gate pro Kategorie: Occupancy, VGPR, LDS, VMEM/SMEM, Wavefronts/SE messen.
- Microbenchmarks pro Kernelgruppe (MMQ, MMVQ, FATTN, VecDot) als Regressionstest.
- Log/Trace: klare Anzeige, ob gfx906-Pfad aktiv ist (Build und Runtime).
