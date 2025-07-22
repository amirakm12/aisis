import re

# Read the file
with open('src/ui/main_window.py', 'r') as f:
    lines = f.readlines()

# Find the last agent import
last_import_index = -1
for i, line in enumerate(lines):
    if line.startswith('from src.agents.style_transfer import StyleTransferAgent'):
        last_import_index = i
        break

if last_import_index != -1:
    new_imports = [
        'from src.agents.vector_image_restoration import VectorImageRestorationAgent\n',
        'from src.agents.vector_style_aesthetic import VectorStyleAestheticAgent\n',
        'from src.agents.vector_denoising import VectorDenoisingAgent\n',
        'from src.agents.vector_text_recovery import VectorTextRecoveryAgent\n',
        'from src.agents.vector_meta_correction import VectorMetaCorrectionAgent\n',
        'from src.agents.vector_semantic_editing import VectorSemanticEditingAgent\n',
        'from src.agents.vector_auto_retouch import VectorAutoRetouchAgent\n',
        'from src.agents.vector_generative import VectorGenerativeAgent\n',
        'from src.agents.vector_neural_radiance import VectorNeuralRadianceAgent\n',
        'from src.agents.vector_super_resolution import VectorSuperResolutionAgent\n',
        'from src.agents.vector_color_correction import VectorColorCorrectionAgent\n',
        'from src.agents.vector_tile_stitching import VectorTileStitchingAgent\n',
        'from src.agents.vector_feedback_loop import VectorFeedbackLoopAgent\n',
        'from src.agents.vector_perspective_correction import VectorPerspectiveCorrectionAgent\n',
        'from src.agents.vector_material_recognition import VectorMaterialRecognitionAgent\n',
        'from src.agents.vector_damage_classifier import VectorDamageClassifierAgent\n',
        'from src.agents.vector_hyperspectral_recovery import VectorHyperspectralRecoveryAgent\n',
        'from src.agents.vector_paint_layer_decomposition import VectorPaintLayerDecompositionAgent\n',
        'from src.agents.vector_self_critique import VectorSelfCritiqueAgent\n',
        'from src.agents.vector_forensic_analysis import VectorForensicAnalysisAgent\n'
    ]
    lines = lines[:last_import_index+1] + new_imports + lines[last_import_index+1:]

# Find the last register
last_register_index = -1
for i, line in enumerate(lines):
    if 'self.orchestrator.register_agent("style_transfer", StyleTransferAgent())' in line:
        last_register_index = i
        break

if last_register_index != -1:
    new_registers = [
        '    self.orchestrator.register_agent("vector_image_restoration", VectorImageRestorationAgent())\n',
        '    self.orchestrator.register_agent("vector_style_aesthetic", VectorStyleAestheticAgent())\n',
        '    self.orchestrator.register_agent("vector_denoising", VectorDenoisingAgent())\n',
        '    self.orchestrator.register_agent("vector_text_recovery", VectorTextRecoveryAgent())\n',
        '    self.orchestrator.register_agent("vector_meta_correction", VectorMetaCorrectionAgent())\n',
        '    self.orchestrator.register_agent("vector_semantic_editing", VectorSemanticEditingAgent())\n',
        '    self.orchestrator.register_agent("vector_auto_retouch", VectorAutoRetouchAgent())\n',
        '    self.orchestrator.register_agent("vector_generative", VectorGenerativeAgent())\n',
        '    self.orchestrator.register_agent("vector_neural_radiance", VectorNeuralRadianceAgent())\n',
        '    self.orchestrator.register_agent("vector_super_resolution", VectorSuperResolutionAgent())\n',
        '    self.orchestrator.register_agent("vector_color_correction", VectorColorCorrectionAgent())\n',
        '    self.orchestrator.register_agent("vector_tile_stitching", VectorTileStitchingAgent())\n',
        '    self.orchestrator.register_agent("vector_feedback_loop", VectorFeedbackLoopAgent())\n',
        '    self.orchestrator.register_agent("vector_perspective_correction", VectorPerspectiveCorrectionAgent())\n',
        '    self.orchestrator.register_agent("vector_material_recognition", VectorMaterialRecognitionAgent())\n',
        '    self.orchestrator.register_agent("vector_damage_classifier", VectorDamageClassifierAgent())\n',
        '    self.orchestrator.register_agent("vector_hyperspectral_recovery", VectorHyperspectralRecoveryAgent())\n',
        '    self.orchestrator.register_agent("vector_paint_layer_decomposition", VectorPaintLayerDecompositionAgent())\n',
        '    self.orchestrator.register_agent("vector_self_critique", VectorSelfCritiqueAgent())\n',
        '    self.orchestrator.register_agent("vector_forensic_analysis", VectorForensicAnalysisAgent())\n'
    ]
    lines = lines[:last_register_index+1] + new_registers + lines[last_register_index+1:]

# Write back
with open('src/ui/main_window.py', 'w') as f:
    f.writelines(lines)
print('Added vector agents to main_window.py')

