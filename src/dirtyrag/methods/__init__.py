from dirtyrag.methods.direct_llm import DirectLLM
from dirtyrag.methods.vanilla_rag import VanillaRAG

METHOD_REGISTRY = {
    DirectLLM.name: DirectLLM,
    VanillaRAG.name: VanillaRAG,
}

