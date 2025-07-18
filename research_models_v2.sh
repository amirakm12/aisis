#!/bin/bash

set -e

models=("RAIM NTIRE 2025 Restore-Any-Image" "Cat-AIR Task-aware all-in-one restoration" "Invert2Restore Zero-shot degradation-blind diffusion" "UniRestore Unified perceptual task-oriented diffusion CVPR 2025" "RestoreVAR Visual autoregressive restoration model" "ZipIR Latent pyramid diffusion ultra-high-res images" "VmambaIR State-space model backbone" "DarkIR All-in-one low-light denoise deblur recovery" "URWKV Multi-state low-light deblur unified model" "InstructRestore Instruction-guided region-specific repair" "TAIR Text-Aware IR preserving visual textual fidelity" "DPIR Dual-Prompting diffusion transformer restoration" "Internal Detail-Preserving Diffusion High-fidelity priors" "Hybrid Transformer-CNN Unified architecture high benchmark" "Restormer Efficient transformer high-res restoration" "SwinIR Swin-Transformer restoration baseline model" "LM4LV LLMRA LLM-driven multi-modal restoration assistants" "AdaptIR Parameter-efficient multi-task adaptation")

for model in "${models[@]}"; do
    echo "Searching for $model"
    curl -s "https://api.github.com/search/repositories?q=$model+image+restoration" | jq -r '.items[0].html_url // "null"'
done