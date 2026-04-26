import re
from typing import Literal
import config
from rag_engine import engine, QueryResult

CONVERSATIONAL_PROMPT = """
You are a helpful, friendly, and intelligent AI assistant.
Your goal is to have a natural, helpful conversation with the user.
Answer questions directly based on your general knowledge.
If the user greets you or asks casual questions, respond warmly and appropriately.
Be concise but complete.
"""


class QueryRouter:
    def __init__(self, fallback_to_chat_on_empty: bool = True):
        self.fallback_to_chat_on_empty = fallback_to_chat_on_empty
        # Update engine's prompt template to use the new RAG_PROMPT
        # Instead of modifying the global template, we dynamically pass it or assume it's set
        # Since we can't easily change the template dynamically without refactoring `generate_rag_response`
        # we will patch the template string at runtime or let it use the global one but updated.
        import rag_engine
        rag_engine.TEXT_QA_TEMPLATE.template = (
            f"{config.SYSTEM_PROMPT}"
            "\n\n"
            "## Retrieved Context\n"
            "{context_str}"
            "\n\n"
            "## Question\n"
            "{query_str}"
            "\n\n"
            "## Answer"
        )

    def classify_query(self, query: str) -> Literal["chat", "rag"]:
        """
        Classifies a query as 'chat' (general conversation) or 'rag' (document grounded).
        Currently uses simple rule-based heuristics. Can be replaced by an LLM classifier.
        """
        query_lower = query.strip().lower()
        
        chat_keywords = {
            "hello", "hi", "hii", "hiii", "hey", "how are you", "who are you", 
            "what can you do", "good morning", "good evening",
            "tell me a joke", "thanks", "thank you", "bye"
        }
        
        if query_lower in chat_keywords:
            return "chat"
            
        for kw in ["hello", "hi ", "hey ", "how are you", "who are you"]:
            if query_lower.startswith(kw):
                return "chat"
        
        return "rag"

    def route_query(self, query: str, temperature: float = 0.7, max_new_tokens: int = 500) -> QueryResult:
        query_type = self.classify_query(query)
        
        if query_type == "chat":
            return self._handle_chat_query(query, temperature, max_new_tokens)
        else:
            return self._handle_rag_query(query, temperature, max_new_tokens)
            
    def _handle_chat_query(self, query: str, temperature: float = 0.7, max_new_tokens: int = 500) -> QueryResult:
        """Handles standard conversational queries without document retrieval."""
        print(f"[router] Routing to CHAT: '{query}'")
        llm_response = ""
        
        if getattr(config, "LLM", None) == "NVIDIA":
            response = engine.nvidia_client.chat.completions.create(
                model=config.NVIDIA_LLM_MODEL,
                messages=[
                    {"role": "system", "content": CONVERSATIONAL_PROMPT},
                    {"role": "user", "content": query}
                ],
                temperature=temperature,
                max_tokens=max_new_tokens,
            )
            llm_response = response.choices[0].message.content
        else:
            full_prompt = f"{CONVERSATIONAL_PROMPT}\n\nUser: {query}\nAnswer:"
            response = engine.mm_llm.complete(prompt=full_prompt, image_documents=[])
            llm_response = str(response).strip()
            
        return QueryResult(answer=llm_response, images=[])

    def _handle_rag_query(self, query: str, temperature: float = 0.7, max_new_tokens: int = 500) -> QueryResult:
        """Handles document-grounded queries using the existing RAG pipeline."""
        print(f"[router] Routing to RAG: '{query}'")
        
        final_text_nodes, final_image_nodes = engine.retrieve_documents(query)
        
        # Handle Empty Retrieval Case
        if not final_text_nodes and not final_image_nodes:
            print("[router] RAG retrieval returned empty context.")
            if self.fallback_to_chat_on_empty:
                print("[router] Falling back to CHAT mode.")
                return self._handle_chat_query(query, temperature, max_new_tokens)
            else:
                return QueryResult(
                    answer="I do not have enough information in the provided documents to answer this question.",
                    images=[]
                )
        
        return engine.generate_rag_response(query, final_text_nodes, final_image_nodes, temperature, max_new_tokens)

# Create a global instance
query_router = QueryRouter()
