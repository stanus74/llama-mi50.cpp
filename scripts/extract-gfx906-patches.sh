#!/bin/bash
# Extrahiert gfx906-Patches nach den 8 Kategorien aus plans/Optimierung.md
# Berücksichtigt zusätzliche Änderungen aus den Dev-Protokollen (Q5_K, etc.)

set -e

# Farben für Output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_DIR="${REPO_ROOT}/patches"
PLANS_DIR="${REPO_ROOT}/plans"

# Baseline Commit (llama.cpp Version ohne deine Änderungen)
# Anpassen falls nötig, oder automatisch ermitteln
if [ -z "$BASELINE_COMMIT" ]; then
    BASELINE_COMMIT="HEAD~10"  # Fallback: 10 Commits zurück
    echo -e "${YELLOW}WARNUNG: BASELINE_COMMIT nicht gesetzt, verwende ${BASELINE_COMMIT}${NC}"
    echo "Setze BASELINE_COMMIT auf den letzten sauberen mainline-Commit"
fi

echo "=========================================="
echo "GFX906 Patch Extraktion"
echo "=========================================="
echo "Repository: ${REPO_ROOT}"
echo "Baseline:   ${BASELINE_COMMIT}"
echo "Output:     ${PATCH_DIR}"
echo ""

# Patch-Verzeichnis erstellen
mkdir -p "${PATCH_DIR}"

# Hilfsfunktion zum Extrahieren eines Patches
extract_patch() {
    local name="$1"
    local files="$2"
    local output="${PATCH_DIR}/${name}"
    
    echo -e "${GREEN}Extrahiere ${name}...${NC}"
    
    if git diff "${BASELINE_COMMIT}" HEAD -- ${files} > "${output}" 2>/dev/null; then
        local lines=$(wc -l < "${output}")
        if [ "$lines" -gt 0 ]; then
            echo "  ✓ ${name} (${lines} Zeilen)"
        else
            echo "  ⚠ ${name} ist leer"
            rm "${output}"
        fi
    else
        echo "  ✗ Fehler bei ${name}"
        rm -f "${output}"
    fi
}

# ============================================
# Kategorie A: MI50/GFX906 Kernel (Herzstück)
# ============================================
echo ""
echo "=========================================="
echo "Kategorie A: gfx906 Kernel (neue Dateien)"
echo "=========================================="

A_FILES="
    ggml/src/ggml-cuda/gfx906/gfx906-common.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-config.h
    ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-fattn-q8.cu
    ggml/src/ggml-cuda/gfx906/gfx906-mmq.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-mmq-prefetch.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_0.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q4_1.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q5_k.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-mmvq-q8_0.cuh
    ggml/src/ggml-cuda/gfx906/gfx906-vecdotq.cuh
"

# Template Instanzen
A_TEMPLATE_FILES=$(find "${REPO_ROOT}/ggml/src/ggml-cuda/gfx906/template-instances" -name "*.cu" 2>/dev/null | sed "s|${REPO_ROOT}/||")

extract_patch "A-gfx906-kernels.patch" "${A_FILES} ${A_TEMPLATE_FILES}"

# ============================================
# Kategorie B: Kernel-Dispatch / Registrierung
# ============================================
echo ""
echo "=========================================="
echo "Kategorie B: Kernel Dispatch"
echo "=========================================="

B_FILES="
    ggml/src/ggml-cuda/mmvq.cu
"

extract_patch "B-gfx906-dispatch.patch" "${B_FILES}"

# ============================================
# Kategorie C: Flash-Attention
# ============================================
echo ""
echo "=========================================="
echo "Kategorie C: Flash-Attention"
echo "=========================================="

C_FILES="
    ggml/src/ggml-cuda/fattn-common.cuh
    ggml/src/ggml-cuda/fattn.cu
"

extract_patch "C-flash-attn-gfx906.patch" "${C_FILES}"

# ============================================
# Kategorie D: Quantization Pipeline
# ============================================
echo ""
echo "=========================================="
echo "Kategorie D: Q8/MXFP4 Pipeline"
echo "=========================================="

D_FILES="
    ggml/src/ggml-cuda/mmq.cuh
    ggml/src/ggml-cuda/vecdotq.cuh
"

extract_patch "D-q8-mxfp4-pipeline.patch" "${D_FILES}"

# ============================================
# Kategorie E: Wave64/MoE Fix
# ============================================
echo ""
echo "=========================================="
echo "Kategorie E: Wave64/MoE Fix"
echo "=========================================="

E_FILES="
    ggml/src/ggml-cuda/mmid.cu
    ggml/src/ggml-cuda/mmq.cu
"

extract_patch "E-wave64-moe-fix.patch" "${E_FILES}"

# ============================================
# Kategorie F: GPU Utility / DPP Warp Utils
# ============================================
echo ""
echo "=========================================="
echo "Kategorie F: DPP Warp Utils"
echo "=========================================="

F_FILES="
    ggml/src/ggml-cuda/common.cuh
    ggml/src/ggml-cuda/ggml-cuda.cu
"

extract_patch "F-dpp-warp-utils.patch" "${F_FILES}"

# ============================================
# Kategorie G: Build-System
# ============================================
echo ""
echo "=========================================="
echo "Kategorie G: Build-System"
echo "=========================================="

G_FILES="
    ggml/src/ggml-cuda/CMakeLists.txt
    ggml/src/ggml-hip/CMakeLists.txt
"

extract_patch "G-gfx906-build.patch" "${G_FILES}"

# ============================================
# Kategorie H: Tooling/Scripts (optional)
# ============================================
echo ""
echo "=========================================="
echo "Kategorie H: Tooling/Scripts"
echo "=========================================="

H_FILES=$(find "${REPO_ROOT}" -maxdepth 1 -name "SCRIPT_*.sh" 2>/dev/null | sed "s|${REPO_ROOT}/||")
if [ -n "$H_FILES" ]; then
    extract_patch "H-mi50-scripts.patch" "${H_FILES}"
else
    echo "  ℹ Keine SCRIPT_*.sh Dateien gefunden"
fi

# ============================================
# Zusammenfassung
# ============================================
echo ""
echo "=========================================="
echo "Zusammenfassung"
echo "=========================================="
echo ""
echo "Extrahierte Patches in ${PATCH_DIR}:"
echo ""

for patch in A-gfx906-kernels.patch B-gfx906-dispatch.patch C-flash-attn-gfx906.patch \
             D-q8-mxfp4-pipeline.patch E-wave64-moe-fix.patch F-dpp-warp-utils.patch \
             G-gfx906-build.patch H-mi50-scripts.patch; do
    if [ -f "${PATCH_DIR}/${patch}" ]; then
        size=$(du -h "${PATCH_DIR}/${patch}" | cut -f1)
        echo "  ✓ ${patch} (${size})"
    else
        echo "  ✗ ${patch} (nicht erstellt)"
    fi
done

echo ""
echo "=========================================="
echo "Nutzung der Patches"
echo "=========================================="
echo ""
echo "1. Patches anwenden (in Reihenfolge):"
echo "   cd /pfad/zu/neuem/llama.cpp"
echo "   git checkout -b gfx906-port"
echo "   for p in F C E G A B D; do"
echo "       git apply --3way patches/\${p}-*.patch || echo \"Konflikt in \${p}\""
echo "   done"
echo ""
echo "2. Bei Konflikten --reject verwenden:"
echo "   git apply --reject patches/A-gfx906-kernels.patch"
echo "   # .rej Dateien manuell bearbeiten"
echo ""
echo "3. Build testen:"
echo "   cmake -B build -DGGML_HIP=ON -DGGML_HIP_GFX906=ON"
echo "   cmake --build build -j"
echo ""

# Statistik
total_size=$(du -sh "${PATCH_DIR}" 2>/dev/null | cut -f1)
echo "Gesamtgröße: ${total_size}"
echo ""
echo -e "${GREEN}Fertig!${NC}"
