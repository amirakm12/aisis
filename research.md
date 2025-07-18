# Research on Aisis Project and Restoration Agents

## Project Structure

- Root contains main.py, launch.py, config files, docs, scripts, etc.
- src/agents/ has base_agent.py and many specialized agents like denoising.py, super_resolution.py.
- Agents inherit from BaseAgent, implement _initialize, _process, _cleanup.
- orchestrator.py initializes all agents and defines a restoration pipeline.
- To add new agents: Create subclass in new .py, import and add to self.agents in orchestrator.py _initialize, add to restoration_pipeline.
- Other dirs: src/core/ with utilities, src/ui/ for interface, plugins/, tests/.

## Restoration Models Research

Searched GitHub for implementations:

- **Cat-AIR**: No direct repo found. Possible paper: Content- and Degradation-Aware Restoration.
- **Invert2Restore**: No repo found. Zero-shot diffusion-based.
- **RAIM (NTIRE 2025)**: No repo found. Benchmark for real-world degradations (upcoming).
- **DarkIR**: https://github.com/cidautai/DarkIR - Low-light restoration.
- **Reelmind**: No repo found. Emerging AI pipeline, details sparse.
- **MambaIRv2**: No direct v2, but MambaIR: https://github.com/GuoShi28/MambaIR - Efficient Mamba-based.
- **ZipIR**: No repo found. Latent Pyramid Diffusion Transformer.
- **DreamClear**: https://github.com/shallowdream204/DreamClear - DiT-based using generative priors.

For integration, will create agent classes with placeholders or clone where possible.

## Next Steps

Create agent files, edit orchestrator.py to include them.