"""
Prompt builder for RAG system
"""

from typing import List, Dict, Any, Optional
import logging


logger = logging.getLogger(__name__)


class PromptBuilder:
    """Builds prompts for RAG generation"""
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        max_context_length: int = 4000,
        include_sources: bool = True,
        instruction_style: str = "detailed"
    ):
        self.system_prompt = system_prompt or self._default_system_prompt()
        self.max_context_length = max_context_length
        self.include_sources = include_sources
        self.instruction_style = instruction_style
    
    async def build_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Build a complete prompt for RAG generation
        """
        try:
            # Step 1: Format context
            formatted_context = self._format_context(context)
            
            # Step 2: Build conversation history
            history_text = self._format_conversation_history(conversation_history)
            
            # Step 3: Create user instruction
            instruction = self._create_instruction(query)
            
            # Step 4: Combine all parts
            prompt_parts = []
            
            if self.system_prompt:
                prompt_parts.append(f"System: {self.system_prompt}")
            
            if history_text:
                prompt_parts.append(f"Conversation History:\n{history_text}")
            
            if formatted_context:
                prompt_parts.append(f"Context:\n{formatted_context}")
            
            prompt_parts.append(f"Question: {query}")
            prompt_parts.append("Answer:")
            
            full_prompt = "\n\n".join(prompt_parts)
            
            # Truncate if too long
            if len(full_prompt) > self.max_context_length:
                full_prompt = self._truncate_prompt(full_prompt, self.max_context_length)
            
            return full_prompt
            
        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            # Fallback to simple prompt
            return f"Question: {query}\n\nAnswer:"
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """
        Format retrieved context documents
        """
        if not context:
            return ""
        
        formatted_docs = []
        
        for i, doc in enumerate(context, 1):
            content = doc.get("content", "").strip()
            if not content:
                continue
            
            doc_text = f"Document {i}:\n{content}"
            
            if self.include_sources:
                metadata = doc.get("metadata", {})
                source = metadata.get("source", "Unknown")
                title = metadata.get("title", "")
                
                if title:
                    doc_text += f"\n(Source: {title} - {source})"
                else:
                    doc_text += f"\n(Source: {source})"
            
            formatted_docs.append(doc_text)
        
        return "\n\n".join(formatted_docs)
    
    def _format_conversation_history(self, history: Optional[List[Dict[str, str]]]) -> str:
        """
        Format conversation history
        """
        if not history:
            return ""
        
        history_parts = []
        for turn in history[-5:]:  # Last 5 turns
            if "user" in turn:
                history_parts.append(f"User: {turn['user']}")
            if "assistant" in turn:
                history_parts.append(f"Assistant: {turn['assistant']}")
        
        return "\n".join(history_parts)
    
    def _create_instruction(self, query: str) -> str:
        """
        Create instruction based on style
        """
        if self.instruction_style == "detailed":
            return """Based on the provided context and conversation history, please provide a comprehensive and accurate answer to the question. 
Use only the information from the context when possible. If the context doesn't contain enough information to answer the question completely, please indicate what information is missing.
Cite your sources when referencing specific information from the documents."""
        
        elif self.instruction_style == "concise":
            return """Based on the context provided, answer the question concisely and accurately. Use only information from the context."""
        
        elif self.instruction_style == "analytical":
            return """Analyze the provided context and provide a detailed answer to the question. 
Consider multiple perspectives if present in the context, and provide a balanced response. 
If there are conflicting pieces of information, acknowledge them and provide the most reasonable interpretation."""
        
        else:
            return f"Answer the following question based on the provided context: {query}"
    
    def _truncate_prompt(self, prompt: str, max_length: int) -> str:
        """
        Truncate prompt while preserving structure
        """
        if len(prompt) <= max_length:
            return prompt
        
        # Try to truncate context section first
        context_start = prompt.find("Context:")
        if context_start != -1:
            context_end = prompt.find("\n\nQuestion:", context_start)
            if context_end != -1:
                # Keep system prompt and history, truncate context
                before_context = prompt[:context_start]
                after_context = prompt[context_end:]
                
                available_space = max_length - len(before_context) - len(after_context)
                if available_space > 100:  # Minimum context to keep
                    context_section = prompt[context_start:context_end]
                    truncated_context = context_section[:available_space - 20] + "..."
                    return before_context + truncated_context + after_context
        
        # Fallback: simple truncation
        return prompt[:max_length - 3] + "..."
    
    def _default_system_prompt(self) -> str:
        """
        Default system prompt
        """
        return """You are a helpful AI assistant that provides accurate, comprehensive answers based on the given context. 
Your role is to:
1. Carefully analyze the provided context documents
2. Synthesize information from multiple sources when relevant
3. Provide clear, well-structured answers
4. Acknowledge when information is insufficient or conflicting
5. Cite sources appropriately when referencing specific information
6. Maintain a professional and helpful tone

Always prioritize accuracy and clarity in your responses."""
    
    async def build_few_shot_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]],
        examples: List[Dict[str, str]]
    ) -> str:
        """
        Build prompt with few-shot examples
        """
        prompt_parts = []
        
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")
        
        # Add examples
        if examples:
            prompt_parts.append("Examples:")
            for i, example in enumerate(examples, 1):
                prompt_parts.append(f"Example {i}:")
                prompt_parts.append(f"Question: {example['question']}")
                prompt_parts.append(f"Answer: {example['answer']}")
                prompt_parts.append("")
        
        # Add current context and question
        formatted_context = self._format_context(context)
        if formatted_context:
            prompt_parts.append(f"Context:\n{formatted_context}")
        
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("Answer:")
        
        return "\n\n".join(prompt_parts)
    
    async def build_chain_of_thought_prompt(
        self,
        query: str,
        context: List[Dict[str, Any]]
    ) -> str:
        """
        Build prompt for chain-of-thought reasoning
        """
        cot_instruction = """Think step-by-step to answer this question:
1. Analyze the question and identify what information is needed
2. Review the provided context for relevant information
3. Synthesize the information to form a coherent answer
4. Provide your final answer with reasoning

Let's work through this step by step."""
        
        formatted_context = self._format_context(context)
        
        prompt_parts = []
        
        if self.system_prompt:
            prompt_parts.append(f"System: {self.system_prompt}")
        
        prompt_parts.append(cot_instruction)
        
        if formatted_context:
            prompt_parts.append(f"Context:\n{formatted_context}")
        
        prompt_parts.append(f"Question: {query}")
        prompt_parts.append("Thought process:")
        
        return "\n\n".join(prompt_parts)
    
    def update_config(self, **kwargs):
        """
        Update prompt builder configuration
        """
        if "system_prompt" in kwargs:
            self.system_prompt = kwargs["system_prompt"]
        if "max_context_length" in kwargs:
            self.max_context_length = kwargs["max_context_length"]
        if "include_sources" in kwargs:
            self.include_sources = kwargs["include_sources"]
        if "instruction_style" in kwargs:
            self.instruction_style = kwargs["instruction_style"]
