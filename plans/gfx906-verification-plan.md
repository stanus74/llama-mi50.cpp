# GFX906 Patch Integration Verification Plan

## Zusammenfassung der Integration

Alle Kategorien A-H aus `Diffs-gfx906-mainline.md` sind im Repository vorhanden:

| Kategorie | Status | Dateien | Beschreibung |
|-----------|--------|---------|--------------|
| **A** | ✅ Integriert | `gfx906/*.cuh`, `gfx906/*.cu` | Neue MI50/GFX906 Kernel |
| **B** | ✅ Integriert | `mmvq.cu` | Dispatch/Registrierung |
| **C** | ✅ Integriert | `fattn.cu`, `fattn-common.cuh` | Flash-Attention Optimierungen |
| **D** | ✅ Integriert | `mmq.cuh`, `vecdotq.cuh` | Quantization Pipeline |
| **E** | ✅ Integriert | `mmid.cu` | Wave64/MoE Korrektheits-Fix |
| **F** | ✅ Integriert | `common.cuh` | DPP Warp Utilities |
| **G** | ✅ Integriert | `CMakeLists.txt` | Build-System Integration |
| **H** | ✅ Integriert | `scripts/` | Tooling/Scripts |

## Build-System Details

### CMakeLists.txt Integration (ggml/src/ggml-hip/CMakeLists.txt)

```cmake
# Zeilen 57-66: gfx906 Dateien hinzufügen
file(GLOB   SRCS "../ggml-cuda/gfx906/*.cuh")
list(APPEND GGML_HEADERS_ROCM ${SRCS})
file(GLOB   GFX906_CU_FILES "../ggml-cuda/gfx906/*.cu")
list(APPEND GGML_SOURCES_ROCM ${GFX906_CU_FILES})
file(GLOB   GFX906_TEMPLATE_FILES "../ggml-cuda/gfx906/template-instances/*.cu")
list(APPEND GGML_SOURCES_ROCM ${GFX906_TEMPLATE_FILES})

# Zeilen 130-132: GFX906 Flag
if (GGML_HIP_GFX906 OR CMAKE_HIP_ARCHITECTURES MATCHES "gfx906")
    add_compile_definitions(GGML_HIP_GFX906)
endif()
```

## Build-Verifikation

### Schritt 1: Konfiguration mit GFX906-Flag

```bash
cmake -B build -DGGML_HIP=ON \
               -DGGML_HIP_GFX906=ON \
               -DCMAKE_HIP_ARCHITECTURES="gfx906" \
               -DGGML_CUDA=OFF
```

**Erwartete Ausgabe:**
```
-- GFX906 optimizations enabled (warp-cooperative Q4/Q5/Q8 MMVQ)
```

### Schritt 2: Template-Instanziierung prüfen

Folgende Dateien müssen kompiliert werden:

| Datei | Template-Instanzen | Zweck |
|-------|-------------------|-------|
| `gfx906-fattn-q8.cu` | `DECL_FATTN_TILE_CASE(64,64)`, `(96,96)`, `(128,128)`, `(256,256)`, `(576,512)` | Flash-Attention Q8 |
| `gfx906-mmvq-q4_0.cuh` | `gfx906_mul_mat_vec_q4_0_warp_coop` | MMVQ Q4_0 |
| `gfx906-mmvq-q4_1.cuh` | `gfx906_mul_mat_vec_q4_1_warp_coop` | MMVQ Q4_1 |
| `gfx906-mmvq-q8_0.cuh` | `gfx906_mul_mat_vec_q8_0_warp_coop` | MMVQ Q8_0 |
| `gfx906-mmvq-q5_k.cuh` | `gfx906_mul_mat_vec_q5_K_warp_coop` | MMVQ Q5_K |

### Schritt 3: Compile-Time Checks

Die folgenden Defines müssen zur Compile-Zeit verfügbar sein:

```cpp
// Aus gfx906-config.h
#define GFX906_FATTN_SPLIT_K_ENABLED 0
#define GFX906_FATTN_N_SPLIT_MAX 1
#define GFX906_MMQ_ITER_K 256
#define GFX906_MMQ_NWARPS 2
#define GFX906_FATTN_Q8_ENABLED 1
#define GFX906_USE_DPP_REDUCTIONS 1
```

### Schritt 4: Dispatcher-Code Verifikation

Der Code in `mmvq.cu` muss den GFX906-Pfad aktivieren:

```cpp
#if defined(GGML_HIP_GFX906)
// GFX906: Use warp-cooperative kernel for ncols_dst=1
if (ncols_dst == 1 && !has_fusion && ncols_x <= 1024) {
    gfx906_launch_mul_mat_vec_q4_0_warp_coop(...);
    break;
}
#endif
```

## Laufzeit-Verifikation

### Umgebungsvariablen

```bash
export HIP_VISIBLE_DEVICES=0
export GGML_HIP_DEBUG=1
```

### Erwartete Kernel-Namen (hipKernelName via rocprof)

Wenn die gfx906-Kernel korrekt geladen werden:

```
gfx906_mul_mat_vec_q4_0_warp_coop
gfx906_mul_mat_vec_q8_0_warp_coop
flash_attn_tile_q8<64,64,...>
flash_attn_tile_q8<128,128,...>
```

## Bekannte Einschränkungen

1. **DKQ=40,80,112**: Temporär deaktiviert (kein Vielfaches von 32)
   - Siehe `fattn-tile-q8-instance-dkq40-dv40.cu`
   - Kommentar: "Phase 5: Temporarily disabled"

2. **GFX906_FATTN_SPLIT_K**: Auf 0 gesetzt (deaktiviert)
   - Vermeidet Overhead bei Prompt-Processing

## Nächste Schritte

1. ✅ Kontext lesen (Diffs-gfx906-mainline.md)
2. ✅ Dateien verifizieren (alle Kategorien A-H vorhanden)
3. ⏳ Build mit `-DGGML_HIP_GFX906` durchführen
4. ⏳ Benchmark vorher/nachher vergleichen
5. ⏳ Integration dokumentieren

---



# GFX906 Patch Integration Verification - COMPLETE

## Status: Alle Kategorien A-H Integriert ✅

### Verifikations-Ergebnis

| Kategorie | Dateien | Status | Nachweis |
|-----------|---------|--------|----------|
| **A** - MI50 Kernel | `gfx906/gfx906-*.cuh`, `gfx906-*.cu` | ✅ | 11 Dateien vorhanden |
| **B** - Dispatch | `mmvq.cu` | ✅ | `gfx906_launch_*` Aufrufe aktiv |
| **C** - Flash-Attention | `fattn.cu`, `fattn-common.cuh` | ✅ | `BEST_FATTN_KERNEL_TILE_Q8` eingebunden |
| **D** - Quantization | `mmq.cuh`, `vecdotq.cuh` | ✅ | `GFX906_MMQ_ITER_K` Konfiguration |
| **E** - Wave64/MoE Fix | `mmid.cu` | ✅ | `n_expert_used >= warp_size / 2` Check |
| **F** - DPP Utils | `common.cuh` | ✅ | `hip_dpp_xor*`, `warp_reduce_amd_f32` |
| **G** - Build-System | `ggml-hip/CMakeLists.txt` | ✅ | `-DGGML_HIP_GFX906` Support |
| **H** - Scripts | `compile.sh`, `Benchmarks.md` | ✅ | Dokumentation vorhanden |

### Build-System Details (ggml/src/ggml-hip/CMakeLists.txt)

```cmake
# Zeilen 63-66: gfx906 Quellen
file(GLOB   GFX906_CU_FILES "../ggml-cuda/gfx906/*.cu")
list(APPEND GGML_SOURCES_ROCM ${GFX906_CU_FILES})
file(GLOB   GFX906_TEMPLATE_FILES "../ggml-cuda/gfx906/template-instances/*.cu")
list(APPEND GGML_SOURCES_ROCM ${GFX906_TEMPLATE_FILES})

# Zeilen 130-132: GFX906 Flag
if (GGML_HIP_GFX906 OR CMAKE_HIP_ARCHITECTURES MATCHES "gfx906")
    add_compile_definitions(GGML_HIP_GFX906)
endif()
```

### Template-Instanzen für fattn-tile-q8

| Instanz | Status | Head-Dim |
|---------|--------|----------|
| `dkq64-dv64` | ✅ | 64 |
| `dkq96-dv96` | ✅ | 96 |
| `dkq128-dv128` | ✅ | 128 |
| `dkq256-dv256` | ✅ | 256 |
| `dkq576-dv512` | ✅ | 576 |
| `dkq40-dv40` | ⚠️ Deaktiviert | 40 (nicht ×32) |
| `dkq80-dv80` | ⚠️ Deaktiviert | 80 (nicht ×32) |
| `dkq112-dv112` | ⚠️ Deaktiviert | 112 (nicht ×32) |

### Nächster Schritt: Build-Verifikation

Um die Integration zu verifizieren:

```bash
# Konfiguration mit GFX906-Optimierungen
cmake -B build -DGGML_HIP=ON \
               -DGGML_HIP_GFX906=ON \
               -DCMAKE_HIP_ARCHITECTURES="gfx906" \
               -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++

# Build
cmake --build build -j$(nproc) 2>&1 | tee build.log

# Verifikation
grep -E "(GFX906 optimizations enabled|gfx906|fattn-tile-q8)" build.log
```

**Erwartete Ausgabe:**
```
-- GFX906 .cu files found: ...
-- GFX906 template instances found: ...
-- GFX906 optimizations enabled (warp-cooperative Q4/Q5/Q8 MMVQ)
```

Die Integration ist vollständig und bereit für den Build-Test.