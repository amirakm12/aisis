import sys
filename = 'src/app_launcher.py'
with open(filename, 'r') as f:
    lines = f.readlines()
# Find import insert point
import_insert = next(i for i, line in enumerate(lines) if 'from .agents.enhanced_image_restoration import EnhancedImageRestorationAgent' in line) + 1
# New imports
new_imports = [
'from .agents.vector_conversion_agent import VectorConversionAgent
',
'from .agents.local_search_agent import LocalSearchAgent
',
'from .agents.global_search_agent import GlobalSearchAgent
',
'from .agents.library_scan_agent import LibraryScanAgent
',
'from .agents.document_analysis_agent import DocumentAnalysisAgent
',
'from .agents.parallel_processing_agent import ParallelProcessingAgent
',
'from .agents.context_match_agent import ContextMatchAgent
',
'from .agents.quality_check_agent import QualityCheckAgent
',
'from .agents.art_history_agent import ArtHistoryAgent
',
'from .agents.crowd_source_agent import CrowdSourceAgent
',
'from .agents.vector_optimization_agent import VectorOptimizationAgent
',
'from .agents.creative_fusion_agent import CreativeFusionAgent
',
'from .agents.trend_sync_agent import TrendSyncAgent
',
'from .agents.image_reconstruction_agent import ImageReconstructionAgent
',
'from .agents.semantic_validation_agent import SemanticValidationAgent
',
'from .agents.speed_blitz_agent import SpeedBlitzAgent
',
'from .agents.style_inference_agent import StyleInferenceAgent
',
'from .agents.vector_creative_agent import VectorCreativeAgent
',
'from .agents.archive_sync_agent import ArchiveSyncAgent
',
'from .agents.user_feedback_agent import UserFeedbackAgent
'
]
lines = lines[:import_insert] + new_imports + lines[import_insert:]
# Find reg insert point
reg_insert = next(i for i, line in enumerate(lines) if 'register_agent("enhanced_restoration", enhanced_restoration)' in line) + 1
# New regs
new_regs = [
'        vector_conversion = VectorConversionAgent()
',
'        register_agent("vector_conversion", vector_conversion)
',
'        local_search = LocalSearchAgent()
',
'        register_agent("local_search", local_search)
',
'        global_search = GlobalSearchAgent()
',
'        register_agent("global_search", global_search)
',
'        library_scan = LibraryScanAgent()
',
'        register_agent("library_scan", library_scan)
',
'        document_analysis = DocumentAnalysisAgent()
',
'        register_agent("document_analysis", document_analysis)
',
'        parallel_processing = ParallelProcessingAgent()
',
'        register_agent("parallel_processing", parallel_processing)
',
'        context_match = ContextMatchAgent()
',
'        register_agent("context_match", context_match)
',
'        quality_check = QualityCheckAgent()
',
'        register_agent("quality_check", quality_check)
',
'        art_history = ArtHistoryAgent()
',
'        register_agent("art_history", art_history)
',
'        crowd_source = CrowdSourceAgent()
',
'        register_agent("crowd_source", crowd_source)
',
'        vector_optimization = VectorOptimizationAgent()
',
'        register_agent("vector_optimization", vector_optimization)
',
'        creative_fusion = CreativeFusionAgent()
',
'        register_agent("creative_fusion", creative_fusion)
',
'        trend_sync = TrendSyncAgent()
',
'        register_agent("trend_sync", trend_sync)
',
'        image_reconstruction = ImageReconstructionAgent()
',
'        register_agent("image_reconstruction", image_reconstruction)
',
'        semantic_validation = SemanticValidationAgent()
',
'        register_agent("semantic_validation", semantic_validation)
',
'        speed_blitz = SpeedBlitzAgent()
',
'        register_agent("speed_blitz", speed_blitz)
',
'        style_inference = StyleInferenceAgent()
',
'        register_agent("style_inference", style_inference)
',
'        vector_creative = VectorCreativeAgent()
',
'        register_agent("vector_creative", vector_creative)
',
'        archive_sync = ArchiveSyncAgent()
',
'        register_agent("archive_sync", archive_sync)
',
'        user_feedback = UserFeedbackAgent()
',
'        register_agent("user_feedback", user_feedback)
'
]
lines = lines[:reg_insert] + new_regs + lines[reg_insert:]
with open(filename, 'w') as f:
    f.writelines(lines)
