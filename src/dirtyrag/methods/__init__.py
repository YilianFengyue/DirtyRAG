from dirtyrag.methods.crag_style_rag import CRAGStyleRAG
from dirtyrag.methods.direct_llm import DirectLLM
from dirtyrag.methods.evidenceboard_rag import EvidenceBoardRAG
from dirtyrag.methods.relevance_filter_rag import RelevanceFilterRAG
from dirtyrag.methods.vanilla_rag import VanillaRAG

METHOD_REGISTRY = {
    DirectLLM.name: DirectLLM,
    VanillaRAG.name: VanillaRAG,
    RelevanceFilterRAG.name: RelevanceFilterRAG,
    CRAGStyleRAG.name: CRAGStyleRAG,
    EvidenceBoardRAG.name: EvidenceBoardRAG,
}
