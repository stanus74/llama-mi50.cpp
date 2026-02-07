# Abweichungsliste (Status pro Hunk)

Stand: 1. Februar 2026

Legende:
- ✅ Integriert (identisch)
- ✅+ Integriert (funktional äquivalent / zusätzliche Erweiterung)
- ⚠️ Abweichung (kosmetisch oder strukturbedingt, funktional gleich)
- ❌ Fehlend

## Patch B — B-gfx906-dispatch.patch
| Hunk | Datei | Änderungstyp | Status | Hinweis |
|---|---|---|---|---|
| B-1 | [ggml/src/ggml-cuda/mmvq.cu](../ggml/src/ggml-cuda/mmvq.cu#L1-L20) | Includes | ✅+ | Zusätzlicher Include `gfx906-mmvq-q5_k.cuh` vorhanden (Erweiterung). |
| B-2 | [ggml/src/ggml-cuda/mmvq.cu](../ggml/src/ggml-cuda/mmvq.cu#L470-L515) | Dispatch Q4_0 | ✅ | GFX906 warp-coop Pfad vorhanden. |
| B-3 | [ggml/src/ggml-cuda/mmvq.cu](../ggml/src/ggml-cuda/mmvq.cu#L515-L550) | Dispatch Q4_1 | ✅ | GFX906 warp-coop Pfad vorhanden. |
| B-4 | [ggml/src/ggml-cuda/mmvq.cu](../ggml/src/ggml-cuda/mmvq.cu#L550-L590) | Dispatch Q8_0 | ✅ | GFX906 warp-coop Pfad vorhanden. |

## Patch C — C-flash-attn-gfx906.patch
| Hunk | Datei | Änderungstyp | Status | Hinweis |
|---|---|---|---|---|
| C-1 | [ggml/src/ggml-cuda/fattn-common.cuh](../ggml/src/ggml-cuda/fattn-common.cuh#L11-L19) | Konstante/Kommentar | ✅ | `FATTN_KQ_MAX_OFFSET` gesetzt. |
| C-2 | [ggml/src/ggml-cuda/fattn-common.cuh](../ggml/src/ggml-cuda/fattn-common.cuh#L270-L282) | Funktion | ✅ | Shuffle auf `ggml_cuda_shfl_xor_sync`. |
| C-3 | [ggml/src/ggml-cuda/fattn-common.cuh](../ggml/src/ggml-cuda/fattn-common.cuh#L920-L936) | Logik | ✅ | `use_stream_k` mit `amd_wmma_available`. |
| C-4 | [ggml/src/ggml-cuda/fattn-common.cuh](../ggml/src/ggml-cuda/fattn-common.cuh#L945-L969) | Logik | ✅ | AMD Split‑K Anpassung. |
| C-5 | [ggml/src/ggml-cuda/fattn.cu](../ggml/src/ggml-cuda/fattn.cu#L1-L20) | Include | ✅ | GFX906 Q8 Kernel Include. |
| C-6 | [ggml/src/ggml-cuda/fattn.cu](../ggml/src/ggml-cuda/fattn.cu#L20-L55) | Logik | ✅ | MMA-Auswahl mit `amd_wmma_available`. |
| C-7 | [ggml/src/ggml-cuda/fattn.cu](../ggml/src/ggml-cuda/fattn.cu#L320-L380) | Logik | ✅ | `gqa_opt_applies` Alignment‑Check. |
| C-8 | [ggml/src/ggml-cuda/fattn.cu](../ggml/src/ggml-cuda/fattn.cu#L270-L520) | Kernel-Auswahl | ✅+ | Q8 Tile Kernel vorhanden; `kernel`-Variable aus Patch nicht nötig (direkter `switch`). |

## Patch D — D-q8-mxfp4-pipeline.patch
| Hunk | Datei | Änderungstyp | Status | Hinweis |
|---|---|---|---|---|
| D-1 | [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L9-L40) | Includes/Konst. | ✅ | GFX906 Includes + MMQ Config. |
| D-2 | [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L270-L295) | Konstante | ✅ | `MMQ_TILE_Y_K_LDS` vorhanden. |
| D-3 | [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L410-L560) | Funktion | ✅ | Vektorisierte Loads Q4_0/Q4_1. |
| D-4 | [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L700-L780) | Funktion/Logik | ✅ | Q8_0 `need_check`-Fix. |
| D-5 | [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L780-L860) | Funktion | ✅ | MXFP4 Software-Pipelining. |
| D-6 | [ggml/src/ggml-cuda/mmq.cuh](../ggml/src/ggml-cuda/mmq.cuh#L3435-L3525) | Funktion | ✅ | Prefetch + LDS-Strides. |
| D-7 | [ggml/src/ggml-cuda/vecdotq.cuh](../ggml/src/ggml-cuda/vecdotq.cuh#L1-L80) | Funktion | ✅ | `get_int_from_mxfp4_table` + GFX906 Pfad. |
| D-8 | [ggml/src/ggml-cuda/vecdotq.cuh](../ggml/src/ggml-cuda/vecdotq.cuh#L80-L140) | Funktion | ✅ | Optimiertes `get_int_from_table_16` (HIP). |
| D-9 | [ggml/src/ggml-cuda/vecdotq.cuh](../ggml/src/ggml-cuda/vecdotq.cuh#L730-L760) | Funktion | ✅ | `gfx906_get_int_b2_fast` verwendet. |
| D-10 | [ggml/src/ggml-cuda/vecdotq.cuh](../ggml/src/ggml-cuda/vecdotq.cuh#L120-L200) | Kommentare | ⚠️ | Patch entfernt einige SIMD-Kommentare; im Repo noch vorhanden. |

## Patch E — E-wave64-moe-fix.patch
| Hunk | Datei | Änderungstyp | Status | Hinweis |
|---|---|---|---|---|
| E-1 | [ggml/src/ggml-cuda/mmid.cu](../ggml/src/ggml-cuda/mmid.cu#L140-L170) | Logik | ✅ | HIP-Fallback bei großen Expert Counts. |
| E-2 | [ggml/src/ggml-cuda/mmq.cu](../ggml/src/ggml-cuda/mmq.cu#L120-L150) | Bugfix | ✅ | `s13 = src1->nb[3] / ts_src1`. |
| E-3 | [ggml/src/ggml-cuda/mmq.cu](../ggml/src/ggml-cuda/mmq.cu#L320-L365) | Logik | ✅ | RDNA3/4 MMQ-Heuristik. |

## Patch F — F-dpp-warp-utils.patch
| Hunk | Datei | Änderungstyp | Status | Hinweis |
|---|---|---|---|---|
| F-1 | [ggml/src/ggml-cuda/common.cuh](../ggml/src/ggml-cuda/common.cuh#L260-L275) | Konstante | ✅ | `LDMATRIX_TRANS_AVAILABLE`. |
| F-2 | [ggml/src/ggml-cuda/common.cuh](../ggml/src/ggml-cuda/common.cuh#L400-L520) | Funktion | ✅ | DPP-Reductions + `ggml_cuda_shfl_xor_sync`. |
| F-3 | [ggml/src/ggml-cuda/common.cuh](../ggml/src/ggml-cuda/common.cuh#L720-L750) | Funktion | ✅ | GFX906 `ggml_cuda_e8m0_to_fp32`. |
| F-4 | [ggml/src/ggml-cuda/common.cuh](../ggml/src/ggml-cuda/common.cuh#L1160-L1230) | Struct/Logik | ✅+ | Graph-Struct erweitert (`disable_due_to_failed_graph_capture`, `cuda_graphs_enabled`). |

## Patch G — G-gfx906-build.patch
| Hunk | Datei | Änderungstyp | Status | Hinweis |
|---|---|---|---|---|
| G-1 | [ggml/src/ggml-cuda/CMakeLists.txt](../ggml/src/ggml-cuda/CMakeLists.txt#L40-L70) | Build-Logik | ✅ | `120a-real 121a-real` konsolidiert. |
