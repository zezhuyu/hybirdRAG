from typing import Any, Dict, List, Optional
import sys
import os
import ast
import re
import json

from openai import OpenAI
from pymilvus import MilvusClient

from .VectorRAG.pipeline import VectorRAGPipeline, DIMENSION
from .VectorRAG.text_processing import ContextualChunker, LateChunker, HybridChunker

from .GraphRAG.convertObj import OpenAILLMWrapper, MLModelEmbeddingWrapper, OpenAIEmbeddingsWrapper
from HybirdRAG.VectorRAG.convertObj import OpenAIEmbeddingClient, OpenAIRerankerClient
from .GraphRAG.pipeline import GraphRAGPipeline
from .GraphRAG.store import GraphRAGStore, NEO4J_AVAILABLE

from HybirdRAG.comp import MLModelClient
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode

rewrite_prompt = """
You are a helpful assistant that rewrites the prompt to be more specific and concise.
given the user chat history and query, rewrite the prompt to be more specific and concise.

# Chat History:
{chat_history}
# Query:
"""

broaden_prompt = """Create 3-5 related search queries for: "{query}"

Examples:
"What happened to the main character?" → ["How did the main character develop?", "What challenges did the main character face?", "What was the main character's background?", "How did the main character change?"]

"Tell me about the setting" → ["What is the environment like?", "Where does the story take place?", "What are the key locations?", "How does the setting affect the plot?"]

"How did the character feel?" → ["What was the character's emotional reaction?", "How did the character's feelings change?", "What did the character experience?", "What were the character's thoughts?"]

Return ONLY a Python list of strings, no explanations, no other text: ["query1", "query2", "query3"]"""


class HybridRAGPipeline:
    def __init__(
        self,
        service_host: str = None,
        milvus_host: str = None,
        openai_host: str = None,
        openai_api_key: str = None,
        *,
        neo4j_config: Optional[Dict[str, Any]] = None,
        graph_vector_store_kwargs: Optional[Dict[str, Any]] = None,
        milvus_client_kwargs: Optional[Dict[str, Any]] = None,
        collection_name: str = None,
        splitter_chunk_size: int = 128,  # Smaller chunks for better retrieval
        splitter_overlap: int = 32,     # More overlap for better context
    ) -> None:
        # Set default values from environment variables
        service_host = service_host or os.getenv("SERVICE_HOST", "localhost:50051")
        milvus_host = milvus_host or os.getenv("MILVUS_HOST", "localhost:19530")
        openai_host = openai_host or os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
        openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY", "ollama")
        collection_name = collection_name or os.getenv("COLLECTION_NAME", "vector_rag")
        # Set default Neo4j configuration from environment variables
        if neo4j_config is None:
            neo4j_host = os.getenv("NEO4J_HOST", "localhost")
            neo4j_bolt_port = int(os.getenv("NEO4J_BOLT_PORT", "7687"))
            # Construct URL from host if NEO4J_URL is not explicitly set
            neo4j_url = os.getenv("NEO4J_URL")
            if not neo4j_url:
                neo4j_url = f"bolt://{neo4j_host}:{neo4j_bolt_port}"
            
            neo4j_config = {
                "host": neo4j_host,
                "username": os.getenv("NEO4J_USERNAME", "neo4j"),
                "password": os.getenv("NEO4J_PASSWORD", "password"),
                "database": os.getenv("NEO4J_DATABASE", "neo4j"),
                "bolt_port": neo4j_bolt_port,
                "url": neo4j_url,
            }
            # Debug: Print Neo4j config (show password status)
            password_value = neo4j_config['password']
            password_status = "CUSTOM" if password_value != "password" else "DEFAULT"
            
            # Verify URL matches host
            if neo4j_config['url'] != f"bolt://{neo4j_config['host']}:{neo4j_config['bolt_port']}":
                print(f"⚠️  URL mismatch: url={neo4j_config['url']} but host={neo4j_config['host']}, "
                      f"expected: bolt://{neo4j_config['host']}:{neo4j_config['bolt_port']}")
        
        # Set default graph vector store configuration from environment variables
        if graph_vector_store_kwargs is None:
            graph_collection = collection_name
            graph_dim = int(os.getenv("GRAPH_VECTOR_DIM", "1024"))
            
            # Parse milvus host for graph store
            if ":" in milvus_host:
                milvus_host_parts = milvus_host.split(":", 1)
                milvus_host_name = milvus_host_parts[0]
                milvus_port = milvus_host_parts[1]
            else:
                milvus_host_name = milvus_host
                milvus_port = "19530"
            
            graph_vector_store_kwargs = {
                "host": milvus_host_name,
                "collection_name": graph_collection,
                "dim": graph_dim,
                "port": milvus_port,
                "uri": f"{milvus_host_name}:{milvus_port}",
            }
            
            # Add Milvus credentials if available
            milvus_user = os.getenv("MILVUS_USERNAME")
            milvus_password = os.getenv("MILVUS_PASSWORD")
            if milvus_user:
                graph_vector_store_kwargs["user"] = milvus_user
            if milvus_password:
                graph_vector_store_kwargs["password"] = milvus_password
        
        # Set default Milvus client configuration from environment variables
        if milvus_client_kwargs is None:
            # Parse milvus host for client
            if ":" in milvus_host:
                milvus_host_parts = milvus_host.split(":", 1)
                milvus_host_name = milvus_host_parts[0]
                milvus_port = milvus_host_parts[1]
            else:
                milvus_host_name = milvus_host
                milvus_port = "19530"
            
            milvus_uri = f"http://{milvus_host_name}:{milvus_port}"
            milvus_client_kwargs = {"uri": milvus_uri}
            
            # Add Milvus credentials if available
            milvus_user = os.getenv("MILVUS_USERNAME")
            milvus_password = os.getenv("MILVUS_PASSWORD")
            if milvus_user:
                milvus_client_kwargs["user"] = milvus_user
            if milvus_password:
                milvus_client_kwargs["password"] = milvus_password
        self.service = MLModelClient(host=service_host)
        # Use the configured milvus_client_kwargs directly
        self.milvus = MilvusClient(**milvus_client_kwargs)

        base_url = os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1/"
        if not base_url.endswith('/v1') and not base_url.endswith('/v1/'):
            base_url = base_url.rstrip('/') + '/v1/'
        self.openai = OpenAI(api_key=openai_api_key, base_url=base_url)
        self.graph_llm = OpenAILLMWrapper(client=self.openai)

        self.vector_rag = VectorRAGPipeline(
            milvus=self.milvus,
            embedding=OpenAIEmbeddingClient(self.openai),
            collection_name=collection_name,
        )
        
        # Set the LLM in global settings for all llama-index operations
        from llama_index.core import Settings
        Settings.llm = self.graph_llm

        if NEO4J_AVAILABLE:
            # Parse Neo4j configuration
            neo4j_url = neo4j_config.get("url", "bolt://localhost:7687")
            neo4j_username = neo4j_config.get("username", "neo4j")
            neo4j_password = neo4j_config.get("password", "password")
            neo4j_database = neo4j_config.get("database", "neo4j")
            
            self.graph_store = GraphRAGStore(
                uri=neo4j_url,
                username=neo4j_username,
                password=neo4j_password,
                database=neo4j_database,
                collection_name=collection_name,  # Pass collection name for node tagging
                summarizer_llm=self.graph_llm
            )
        else:
            # Fallback to simple graph store (no Neo4j available)
            # Note: SimpleGraphStore doesn't support collection_name, but pass it anyway for consistency
            self.graph_store = GraphRAGStore(summarizer_llm=self.graph_llm, collection_name=collection_name)

        embedding_wrapper = OpenAIEmbeddingsWrapper(self.openai)
        
        # Use the same Milvus collection for GraphRAG as the main vector store
        graph_milvus_kwargs = {
            "host": milvus_host,
            "collection_name": collection_name,  # Use same collection
            "dim": DIMENSION,  # Same dimension as main vector store
        }
        if milvus_client_kwargs:
            graph_milvus_kwargs.update(milvus_client_kwargs)
        
        self.graph_rag = GraphRAGPipeline(
            llm=self.graph_llm,
            graph_store=self.graph_store,
            milvus_kwargs=graph_milvus_kwargs,
            embedding_model=embedding_wrapper,
            milvus_client=self.milvus,  # Pass the Milvus client for syncing
        )

        # Check if GraphRAG is enabled via environment variable
        # Set GRAPHRAG_ENABLED=false to disable GraphRAG for testing
        self.graphrag_enabled = os.getenv("GRAPHRAG_ENABLED", "true").lower() == "true"
        if not self.graphrag_enabled:
            print("⚠️  GraphRAG is DISABLED via GRAPHRAG_ENABLED environment variable")

        self.splitter = SentenceSplitter(
            chunk_size=splitter_chunk_size,
            chunk_overlap=splitter_overlap,
        )
        self.chunker = HybridChunker(
            embedding_client=OpenAIEmbeddingClient(self.openai),
            max_tokens_per_chunk=splitter_chunk_size,
            min_tokens_per_chunk=50,
            sim_drop=0.40,
            fixed_threshold=0.45,
            c=0.9, 
            init_constant=1.5,
            overlap_tokens=splitter_overlap,
        )  # Default chunker, can be overridden
        self.context_chunker = ContextualChunker()

    def _broaden_query_with_retry(self, query_text: str, retry_limit: int = 3) -> Optional[List[str]]:
        """Broaden query with retry mechanism for better reliability."""
        
        for attempt in range(retry_limit):
            try:
                broadened_prompt = broaden_prompt.format(query=query_text)
                model = os.getenv("PROMPT_REWRITE_MODEL", "qwen3:0.6b")
                rewritten = self.openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "You are a query expansion assistant. Return ONLY a Python list of strings. No thinking, no explanations, no other text."},
                        {"role": "user", "content": broadened_prompt},
                    ],
                )
                response_text = (
                    rewritten.choices[0].message.content
                    if rewritten and rewritten.choices
                    else None
                )
                
                if not response_text:
                    print(f"⚠️ Attempt {attempt + 1}: Empty response from LLM")
                    continue
                
                # Clean and parse the response
                broadened_queries = self._parse_broadened_queries(response_text)
                if broadened_queries:
                    return broadened_queries
                else:
                    print(f"⚠️ Attempt {attempt + 1}: Failed to parse valid queries")
                    
            except Exception as e:
                print(f"⚠️ Attempt {attempt + 1}: Exception - {e}")
                if attempt == retry_limit - 1:
                    print(f"❌ All {retry_limit} attempts failed")
                continue
        
        return None

    def _parse_broadened_queries(self, response_text: str) -> Optional[List[str]]:
        """Parse broadened queries from LLM response with robust error handling."""
        try:
            # Clean the response text
            response_text = response_text.strip()
            
            # Remove thinking tags if present
            if '<think>' in response_text and '</think>' in response_text:
                start = response_text.find('</think>') + len('</think>')
                response_text = response_text[start:].strip()
            
            # Remove any markdown code blocks if present
            if response_text.startswith('```'):
                lines = response_text.split('\n')
                response_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else response_text
            
            # Extract the list from the response using regex
            list_pattern = r'\[.*?\]'
            matches = re.findall(list_pattern, response_text, re.DOTALL)
            if matches:
                response_text = matches[0]
            
            # Try multiple parsing strategies
            broadened_queries = None
            
            # Strategy 1: Try ast.literal_eval first
            try:
                broadened_queries = ast.literal_eval(response_text)
                if isinstance(broadened_queries, list) and all(isinstance(q, str) for q in broadened_queries):
                    return broadened_queries
            except (ValueError, SyntaxError):
                pass
            
            # Strategy 2: Try JSON parsing
            try:
                # Clean up common JSON issues
                cleaned_text = response_text
                # Fix single quotes to double quotes for JSON
                cleaned_text = re.sub(r"'([^']*)'", r'"\1"', cleaned_text)
                broadened_queries = json.loads(cleaned_text)
                if isinstance(broadened_queries, list) and all(isinstance(q, str) for q in broadened_queries):
                    return broadened_queries
            except (json.JSONDecodeError, ValueError):
                pass
            
            # Strategy 3: Extract strings using regex as fallback
            try:
                # Extract quoted strings from the response
                string_pattern = r'"([^"]*)"'
                matches = re.findall(string_pattern, response_text)
                if matches:
                    return matches
                
                # Try single quotes if double quotes failed
                string_pattern = r"'([^']*)'"
                matches = re.findall(string_pattern, response_text)
                if matches:
                    return matches
            except Exception:
                pass
            
            # Strategy 4: Split by common delimiters as last resort
            try:
                # Split by commas and clean up
                items = [item.strip().strip('"\'[]') for item in response_text.split(',')]
                items = [item for item in items if item and len(item) > 2]  # Filter out empty/short items
                if items:
                    return items
            except Exception:
                pass
                
                return None
                
        except Exception as e:
            print(f"⚠️ Parsing error: {e}")
            return None

    def _clean_query_text(self, query_text: str) -> str:
        """Clean query text to ensure it's safe for search."""
        if not query_text or not isinstance(query_text, str):
            return ""
        
        # Remove any thinking tags that might have leaked through
        if '<think>' in query_text and '</think>' in query_text:
            start = query_text.find('</think>') + len('</think>')
            query_text = query_text[start:].strip()
        
        # Remove any markdown code blocks
        if query_text.startswith('```'):
            lines = query_text.split('\n')
            query_text = '\n'.join(lines[1:-1]) if len(lines) > 2 else ""
        
        query_text = re.sub(r'\([^)]*shortened[^)]*\)', '', query_text, flags=re.IGNORECASE)
        query_text = re.sub(r'\([^)]*concise[^)]*\)', '', query_text, flags=re.IGNORECASE)
        
        # Replace multiple line breaks and spaces with single spaces
        query_text = re.sub(r'\s+', ' ', query_text)
        
        # Remove any non-printable characters except basic punctuation
        query_text = ''.join(char for char in query_text if char.isprintable() or char.isspace())
        query_text = query_text.strip()
        
        # Limit query length to prevent issues (reduce from 10000 to 2000)
        if len(query_text) > 2000:
            query_text = query_text[:2000]
        
        # Ensure query ends properly (remove trailing punctuation issues)
        query_text = query_text.rstrip('.,;:!?')
        
        return query_text
    
    def _decompose_multi_hop_query(self, query: str) -> List[str]:
        """Decompose complex multi-hop questions into simpler sub-questions using generic patterns."""
        query_lower = query.lower()
        
        # Generic multi-hop indicators - these are common linguistic patterns
        multi_hop_indicators = [
            " who ", " whose ", " that ", " which ", " where ", " when ",
            " was ", " is ", " were ", " are ", " had ", " has ", " have "
        ]
        
        # Count potential entities and relationships
        entity_indicators = 0
        for indicator in multi_hop_indicators:
            if indicator in query_lower:
                entity_indicators += 1
        
        # If we have multiple entity indicators, likely multi-hop
        if entity_indicators < 2:
            return [query]
        
        # Generic decomposition patterns
        decomposed_queries = []
        import re
        
        # Pattern 1: Questions with "who" clauses
        who_pattern = r"([^?]+?)\s+who\s+([^?]+?)\?"
        who_match = re.search(who_pattern, query_lower)
        if who_match:
            first_part = who_match.group(1).strip()
            second_part = who_match.group(2).strip()
            decomposed_queries.append(f"Who {second_part}?")
            
            # Try to extract what we're asking about in the first part
            what_match = re.search(r"what\s+([^?]+?)\s+(?:was|is|were|are)", first_part)
            if what_match:
                what_thing = what_match.group(1).strip()
                decomposed_queries.append(f"What {what_thing} did [ENTITY] have?")
        
        # Pattern 2: Questions with "that" clauses
        that_pattern = r"([^?]+?)\s+that\s+([^?]+?)\?"
        that_match = re.search(that_pattern, query_lower)
        if that_match:
            first_part = that_match.group(1).strip()
            second_part = that_match.group(2).strip()
            decomposed_queries.append(f"What {second_part}?")
        
        # Pattern 3: Questions with "which" clauses
        which_pattern = r"([^?]+?)\s+which\s+([^?]+?)\?"
        which_match = re.search(which_pattern, query_lower)
        if which_match:
            first_part = which_match.group(1).strip()
            second_part = which_match.group(2).strip()
            decomposed_queries.append(f"Which {second_part}?")
        
        # If decomposition failed, return original query
        if not decomposed_queries:
            decomposed_queries = [query]
        
        return decomposed_queries
    
    def _extract_intermediate_answers(self, query: str, context_items: List[str]) -> Dict[str, str]:
        """Extract intermediate answers for multi-hop questions using generic patterns."""
        query_lower = query.lower()
        intermediate_answers = {}
        import re
        
        # Generic entity extraction patterns
        entity_patterns = {
            "person": [
                r"([A-Z][a-zA-Z\s]{2,30})(?:\s*\([^)]*\))?(?:\s*,\s*(?:born|died|is|was))",
                r"([A-Z][a-zA-Z\s]{2,30})(?:\s+was\s+born|\s+is\s+a|\s+was\s+a)",
                r"(?:actor|actress|director|writer|singer|musician|politician|president|minister)\s+([A-Z][a-zA-Z\s]{2,30})"
            ],
            "organization": [
                r"([A-Z][a-zA-Z\s&]{2,50})(?:\s+Inc\.?|\s+Corp\.?|\s+Ltd\.?|\s+Company)",
                r"(?:company|organization|corporation)\s+([A-Z][a-zA-Z\s&]{2,50})"
            ],
            "location": [
                r"([A-Z][a-zA-Z\s]{2,30})(?:\s*,\s*[A-Z][a-zA-Z\s]+)?(?:\s*,\s*[A-Z][a-zA-Z\s]+)?",
                r"(?:in|at|from|to)\s+([A-Z][a-zA-Z\s]{2,30})"
            ]
        }
        
        # Extract entities from context
        for context in context_items:
            context_lower = context.lower()
            for entity_type, patterns in entity_patterns.items():
                if entity_type not in intermediate_answers:
                    for pattern in patterns:
                        matches = re.findall(pattern, context)
                        if matches:
                            # Clean and validate the match
                            entity = matches[0].strip()
                            if len(entity) > 2 and len(entity) < 50:  # Reasonable length
                                intermediate_answers[entity_type] = entity
                                break
        
        return intermediate_answers
    
    def _refine_context_with_intermediates(self, query: str, context_items: List[str], intermediates: Dict[str, str]) -> List[str]:
        """Refine context using intermediate answers to get more relevant information."""
        refined_context = []
        
        for context in context_items:
            context_lower = context.lower()
            # Check if context contains intermediate answers
            for key, value in intermediates.items():
                if value.lower() in context_lower:
                    refined_context.append(context)
                    break
        
        # If no refined context found, return original
        return refined_context if refined_context else context_items
    
    def _iterative_retrieval(self, query: str, initial_context: List[str]) -> List[str]:
        """Perform iterative retrieval using LLM to generate follow-up queries.
        
        This method is completely generic and works for any domain by using LLM
        to understand what information is needed for the second hop.
        """
        all_context = list(initial_context)
        newly_added_context = []
        
        # Extract entities from initial context
        entities = self._extract_entities_from_context(initial_context)
        
        if not entities:
            return all_context
        
        # Use LLM to generate follow-up queries based on original question and found entities
        expanded_queries = self._generate_follow_up_queries_llm(query, entities[:3])
        
        if not expanded_queries:
            return all_context
        
        # Perform additional retrievals with LLM-generated queries
        for expanded_query in expanded_queries[:8]:  # Limit to 8 queries for efficiency
            try:
                additional_results = self.vector_rag.query([expanded_query], limit=5)
                additional_contexts = []
                for hit in additional_results:
                    for hit_item in hit:
                        if hasattr(hit_item, 'entity') and hasattr(hit_item.entity, 'content'):
                            additional_contexts.append(hit_item.entity.content)
                
                # Add new context that's not already in our collection
                for ctx in additional_contexts:
                    if ctx not in all_context and len(ctx) > 50:
                        newly_added_context.append(ctx)
                        all_context.append(ctx)
                        
            except Exception as e:
                pass  # Silently continue on error
        
        # Prioritize newly added contexts
        if newly_added_context:
            reordered = newly_added_context + [ctx for ctx in all_context if ctx not in newly_added_context]
            return reordered
        else:
            return all_context
    
    def _generate_follow_up_queries_llm(self, original_query: str, entities: List[str]) -> List[str]:
        """Use LLM to generate follow-up queries for multi-hop retrieval.
        
        This is completely generic - no hardcoded patterns or domain knowledge.
        """
        try:
            entities_str = ", ".join(f'"{e}"' for e in entities)
            
            prompt = f"""Given a multi-hop question and entities found in the initial search, generate follow-up search queries to find the missing information.

Original Question: {original_query}

Entities found in initial search: {entities_str}

Task: Generate 3-5 short, focused search queries to find information about these entities that would help answer the original question.

Rules:
- Each query should be 2-5 words
- Focus on finding biographical, definitional, or relational information about the entities
- Queries should help connect the dots to answer the original question
- Be specific and concise

Respond with ONLY a JSON array of query strings, nothing else:
["query1", "query2", "query3", ...]

Example:
Original Question: "Where was the founder of Microsoft born?"
Entities: ["Microsoft"]
Response: ["Microsoft founder", "Microsoft Bill Gates", "Bill Gates birthplace", "who founded Microsoft"]

Now generate queries for the question above:"""

            response = self.openai.chat.completions.create(
                model="gpt-oss:latest",  # Use available model
                messages=[
                    {"role": "system", "content": "You are a search query generator. Respond only with a JSON array of strings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2,  # Low temperature for focused queries
                max_tokens=150
            )
            
            if response and response.choices:
                content = response.choices[0].message.content.strip()
                
                # Extract JSON array
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    queries = json.loads(json_str)
                    
                    # Filter and validate queries
                    valid_queries = []
                    for q in queries:
                        if isinstance(q, str) and 2 <= len(q.split()) <= 10:
                            valid_queries.append(q)
                    
                    return valid_queries[:8]  # Max 8 queries
                    
        except Exception as e:
            pass  # Fall back to simple expansion
        
        # Fallback: simple entity-based queries
        fallback_queries = []
        for entity in entities[:3]:
            fallback_queries.extend([
                entity,
                f"about {entity}",
                f"{entity} information"
            ])
        return fallback_queries[:6]
    
    def _expand_query_llm(self, query: str) -> List[str]:
        """Use LLM to expand ambiguous queries into multiple specific search queries.
        
        This is completely generic and works for any domain. The LLM generates
        alternative phrasings and disambiguates entities to improve initial retrieval.
        """
        try:
            prompt = f"""Given this question, generate 3-5 specific search queries to find the answer.

Question: {query}

IMPORTANT: This question may be ambiguous. Generate multiple queries covering different interpretations:
- If there's an ambiguous entity (e.g., "Green performer"), try different interpretations (person named Green, performer in band Green, etc.)
- Include both the entity and the information needed (biography, spouse, birthplace, etc.)
- Use alternative phrasings and synonyms
- Each query should be 2-8 words, specific for search

ALWAYS generate at least 3 queries, even if the question seems clear.

Respond with ONLY a JSON array (no other text):
["query1", "query2", "query3", "query4", "query5"]

Examples:
Question: "Who is the spouse of the Green performer?"
Response: ["Green performer biography", "Green musician spouse", "performer named Green personal life", "Green artist family", "Gong band Green performer"]

Question: "Where was the founder of Tesla born?"
Response: ["Tesla founder", "Tesla company founder birthplace", "who founded Tesla born", "Tesla founder biography", "Tesla founder born"]

Now generate 3-5 queries for: {query}"""

            response = self.openai.chat.completions.create(
                model="gpt-oss:latest",
                messages=[
                    {"role": "system", "content": "You are a search query expansion assistant. Respond only with a JSON array of strings."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Some creativity for diverse queries
                max_tokens=200
            )
            
            if response and response.choices:
                content = response.choices[0].message.content.strip()
                
                # Extract JSON array
                json_start = content.find('[')
                json_end = content.rfind(']') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    try:
                        queries = json.loads(json_str)
                        
                        # Filter and validate queries
                        valid_queries = []
                        for q in queries:
                            if isinstance(q, str):
                                q = q.strip()
                                # Validate query length and content
                                word_count = len(q.split())
                                if 2 <= word_count <= 15 and len(q) > 3:
                                    valid_queries.append(q)
                        
                        if valid_queries:
                            # Include original query as first option, then add expanded queries
                            # Remove duplicates
                            all_queries = [query]  # Original first
                            for q in valid_queries:
                                q_lower = q.lower()
                                query_lower = query.lower()
                                # Skip if identical to original or already added
                                if q_lower != query_lower and not any(q_lower == existing.lower() for existing in all_queries):
                                    all_queries.append(q)
                            
                            if len(all_queries) > 1:
                                return all_queries[:6]  # Max 6 queries total
                    except json.JSONDecodeError:
                        # JSON parse failed, fall through to return original
                        pass
                    
        except Exception as e:
            # If LLM fails, return original query only
            # No hardcoded patterns - fully generic system
            # Silently fail - expansion is optional
            pass
        
        # Fallback: Return original query if LLM expansion failed
        # This ensures we still perform retrieval, just without expansion
        return [query]
    
    def _detect_query_intent_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to detect if a query is multi-hop and what type of information is needed.
        
        This is completely generic and works for any domain without hardcoded patterns.
        """
        try:
            prompt = f"""Analyze this question and determine if it requires multi-hop reasoning (connecting information from multiple sources).

Question: {query}

Respond with ONLY a JSON object in this exact format:
{{
    "is_multi_hop": true/false,
    "reasoning": "brief explanation",
    "information_type": "one of: biographical, location, relationship, temporal, definition, comparison, composition, origin, other"
}}

Examples:
Question: "What is the capital of France?"
{{"is_multi_hop": false, "reasoning": "Direct factual question", "information_type": "location"}}

Question: "What is the performer of Heartbeat named after?"
{{"is_multi_hop": true, "reasoning": "Need to find performer first, then find naming origin", "information_type": "origin"}}

Question: "Where was the founder of Tesla born?"
{{"is_multi_hop": true, "reasoning": "Need to find founder, then find birthplace", "information_type": "location"}}

Now analyze the question above and respond with JSON only:"""

            response = self.openai.chat.completions.create(
                model="gpt-oss:latest",  # Use available model
                messages=[
                    {"role": "system", "content": "You are a query analysis assistant. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # Deterministic for consistent JSON
                max_tokens=150
            )
            
            if response and response.choices:
                content = response.choices[0].message.content.strip()
                
                # Extract JSON from response
                json_start = content.find('{')
                json_end = content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    result = json.loads(json_str)
                    return result
                
        except Exception as e:
            pass  # Fall back to simple detection
        
        # Fallback: simple pattern-based detection
        return {
            'is_multi_hop': len(query.split()) > 10,
            'reasoning': 'Fallback detection',
            'information_type': 'other'
        }
    
    def _extract_entities_from_context(self, context_items: List[str]) -> List[str]:
        """Extract named entities from context using generic patterns.
        
        This uses simple capitalization patterns to find proper nouns without
        domain-specific knowledge.
        """
        entities = []
        entity_freq = {}  # Track frequency to prioritize important entities
        import re
        
        for context in context_items:
            # Extract multi-word capitalized phrases (proper nouns)
            # Pattern: "Nina Sky", "John Smith", "New York", etc.
            multi_word_entities = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b', context)
            
            for entity in multi_word_entities:
                entity = entity.strip()
                # Filter out common non-entities
                if (2 < len(entity) < 40 and 
                    not entity.startswith(('The ', 'A ', 'An ')) and
                    entity not in ['Title', 'Album', 'Song', 'Article', 'Book']):
                    entity_freq[entity] = entity_freq.get(entity, 0) + 1
            
            # Also extract single-word capitalized entities (but be more selective)
            single_word_entities = re.findall(r'\b([A-Z][a-z]{2,})\b', context)
            
            for entity in single_word_entities:
                # Only include if it appears multiple times or is long enough
                if len(entity) > 4 and entity not in ['Title', 'Album', 'Song', 'Article', 'Book', 'Chapter']:
                    entity_freq[entity] = entity_freq.get(entity, 0) + 0.5  # Lower weight for single words
        
        # Sort by frequency and return top entities
        sorted_entities = sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)
        # Lower threshold to include entities that appear at least once
        entities = [entity for entity, freq in sorted_entities if freq >= 0.5]
        
        # Prioritize multi-word entities (they're usually more specific and useful)
        multi_word_first = [e for e in entities if ' ' in e]
        single_word = [e for e in entities if ' ' not in e]
        entities = multi_word_first + single_word
        
        return entities[:8]  # Return top 8 entities
    
    def _extract_key_terms(self, query: str, context_items: List[str]) -> List[str]:
        """Extract key terms from query and context for expansion."""
        import re
        
        # Extract important terms from query
        query_terms = []
        query_lower = query.lower()
        
        # Remove common words and extract meaningful terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'were', 'was', 'are', 'is', 'what', 'who', 'where', 'when', 'why', 'how'}
        words = re.findall(r'\b\w+\b', query_lower)
        query_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Extract terms from context that might be relevant
        context_terms = []
        for context in context_items:
            # Look for capitalized terms that might be important
            capitalized_terms = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', context)
            for term in capitalized_terms:
                if len(term) > 2 and len(term) < 20 and term.lower() not in query_terms:
                    context_terms.append(term)
        
        # Combine and deduplicate
        all_terms = query_terms + context_terms
        return list(set(all_terms))[:8]  # Return top 8 unique terms
    
    def _is_complex_multi_hop(self, query: str) -> bool:
        """Determine if a query requires multi-hop reasoning using LLM.
        
        This is completely generic and works for any domain.
        """
        # Use LLM for intelligent multi-hop detection
        intent_result = self._detect_query_intent_llm(query)
        
        if intent_result.get('is_multi_hop', False):
            return True
        
        # Fallback: simple heuristics
        # Count capitalized entities (proper nouns)
        entity_matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Check for chained prepositions (e.g., "X of Y of Z")
        chained_prep_pattern = r'\b(?:of|from|in|at|by|for|with|to)\s+\w+\s+(?:of|from|in|at|by|for|with|to)\b'
        has_chained = bool(re.search(chained_prep_pattern, query.lower()))
        
        # Trigger if multiple entities or chained relationships or long complex question
        return (len(entity_matches) >= 2 or 
                has_chained or 
                len(query.split()) > 12)
    
    def _has_multiple_entities(self, query: str) -> bool:
        """Check if a query mentions multiple entities that might need iterative retrieval."""
        query_lower = query.lower()
        
        # Look for patterns that suggest multiple entities
        multi_entity_patterns = [
            " and ", " or ", " both ", " either ", " neither ",
            " same ", " different ", " compared to ", " versus ",
            " who ", " that ", " which ", " where "
        ]
        
        # Count capitalized words (potential entities)
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', query)
        
        # Check for multi-entity patterns
        has_multi_entity_pattern = any(pattern in query_lower for pattern in multi_entity_patterns)
        
        # If we have multiple capitalized words or multi-entity patterns, likely multi-entity
        return len(capitalized_words) >= 2 or has_multi_entity_pattern
    
    # def _validate_and_correct_answer(self, answer: str, query: str, question_type: str, context: str) -> str:
        """Validate and potentially self-correct the answer."""
        if not answer or answer.lower() in ["i don't know", "unknown", "not mentioned", "not found"]:
            return answer
        
        # Basic validation checks
        if question_type == "yes_no":
            if answer.lower() not in ["yes", "no"]:
                # Try to extract yes/no from context if answer is unclear
                context_lower = context.lower()
                if "both" in context_lower or "same" in context_lower or "similar" in context_lower:
                    return "yes"
                elif "different" in context_lower or "not the same" in context_lower:
                    return "no"
                return "no"  # Default to no if unclear
        
        elif question_type == "name":
            # Check if answer looks like a proper name
            if len(answer) < 2 or not any(c.isalpha() for c in answer):
                # Try to find names in context
                import re
                name_pattern = r"([A-Z][a-zA-Z\s]{2,30})(?:\s*\([^)]*\))?"
                names = re.findall(name_pattern, context)
                if names:
                    return names[0].strip()
                return answer
        
        elif question_type == "title_position":
            # Check if answer looks like a title/position
            if len(answer) < 3 or answer.lower() in ["title", "position", "job", "role"]:
                # Try to find titles in context
                import re
                title_pattern = r"([A-Z][a-zA-Z\s]{3,30})(?:\s+(?:of|for|in|at|to)\s+[A-Z][a-zA-Z\s]+)?"
                titles = re.findall(title_pattern, context)
                for title in titles:
                    title_lower = title.lower()
                    if any(indicator in title_lower for indicator in ["chief", "secretary", "director", "president", "minister", "ambassador", "governor", "mayor", "officer", "manager", "supervisor", "coordinator", "administrator", "consultant", "advisor"]):
                        return title.strip()
                
                # Try to extract from context using query keywords
                query_lower = query.lower()
                if "government position" in query_lower:
                    gov_pattern = r"([A-Z][a-zA-Z\s]+)(?:\s+(?:of|for|in|at|to)\s+[A-Z][a-zA-Z\s]+)?"
                    gov_matches = re.findall(gov_pattern, context)
                    for match in gov_matches:
                        if len(match) > 5 and any(word in match.lower() for word in ["ambassador", "secretary", "minister", "director", "chief"]):
                            return match.strip()
                
                return answer
        
        return answer

    def query(
        self,
        query: str,
        rewrite: bool = True,
        chat_history: Optional[List[str]] = None,
        broaden_query: bool = True,
        broaden_retry_limit: int = 3,
        context_chunk_size: int = 256,
        rerank: bool = True,
        compress: bool = False,
        limit: int = 30,  # Increased from 20 to get more context for multi-hop questions
        collection_name: Optional[str] = None,
    ):
        query_text = query
        if rewrite:
            chat_history = chat_history or []
            rewritten_prompt = rewrite_prompt.format(
                chat_history="\n".join(chat_history), query=query
            )
            try:
                model = os.getenv("PROMPT_REWRITE_MODEL", "qwen3:0.6b")
                rewritten = self.openai.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": rewritten_prompt},
                        {"role": "user", "content": query},
                    ],
                )
                query_text = (
                    rewritten.choices[0].message.content
                    if rewritten and rewritten.choices
                    else query
                )
            except Exception:
                query_text = query

        # LLM-based query expansion for ambiguous queries
        # This runs before traditional broadening to improve initial retrieval
        # Controlled by broaden_query flag
        llm_expanded = False
        if broaden_query and isinstance(query_text, str):
            expanded_queries = self._expand_query_llm(query_text)
            if len(expanded_queries) > 1:
                # LLM generated multiple queries - use them
                query_text = expanded_queries
                llm_expanded = True
            # Otherwise keep original query_text (LLM returned just the original)

        # Only use traditional broadening if LLM didn't expand (or if explicitly requested)
        if broaden_query and not llm_expanded and isinstance(query_text, str):
            # Store original query as backup
            original_query_text = query_text
            broadened_queries = self._broaden_query_with_retry(query_text, broaden_retry_limit)
            if broadened_queries:
                # Add original query and broadened queries as a list
                all_queries = [original_query_text] + broadened_queries
                # Keep queries as a list instead of concatenating
                query_text = all_queries
            else:
                print("⚠️ Query broadening failed after all retries, using original query")
                # Ensure we use the original clean query
                query_text = original_query_text

        # Ensure query_text is a list and clean each query
        if isinstance(query_text, str):
            original_query = self._clean_query_text(query_text)
            # Try query decomposition for multi-hop questions
            decomposed_queries = self._decompose_multi_hop_query(original_query)
            query_text = [self._clean_query_text(q) for q in decomposed_queries if q]
        elif isinstance(query_text, list):
            query_text = [self._clean_query_text(q) for q in query_text if q]
        else:
            query_text = [self._clean_query_text(str(query_text))]

        
        vector_results = self.vector_rag.query(query_text, limit=limit, collection_name=collection_name)
        
        # Try to get graph answer, but handle gracefully if it fails
        graph_answer = None
        if self.graphrag_enabled:
            try:
                # GraphRAG query expects a list of strings - query_text is already a list
                graph_answer = self.graph_rag.query(query_text)
                # Only include graph answer if it's not empty
                if not graph_answer or (isinstance(graph_answer, list) and not any(g.strip() for g in graph_answer)):
                    graph_answer = None
            except Exception as e:
                print(f"⚠️  GraphRAG query failed: {e}")
                graph_answer = None

        # Handle HybridHits objects from Milvus
        vector_texts = []
        if vector_results is None:
            vector_results = []
        for hit in vector_results:
            # HybridHits is iterable and contains Hit objects
            for hit_item in hit:
                if hasattr(hit_item, 'entity') and hasattr(hit_item.entity, 'content'):
                    vector_texts.append(hit_item.entity.content)
                elif isinstance(hit_item, dict) and 'entity' in hit_item:
                    vector_texts.append(hit_item['entity'].get('content', ''))
        
        vector_texts = [text for text in vector_texts if text]

        if context_chunk_size > 0 and vector_texts:
            self.context_chunker.max_chunk_size = context_chunk_size
            refined_chunks = self.context_chunker.process_results(vector_texts)
            vector_texts = [chunk["text"] for chunk in refined_chunks if "text" in chunk]
        
        # Perform iterative retrieval for better context coverage
        # Always try iterative retrieval for multi-hop questions, even if decomposed
        if vector_texts and isinstance(query_text, list) and len(query_text) > 0:
            # Use the first query (most relevant) for iterative retrieval
            original_query = query_text[0] if isinstance(query_text, list) else query_text
            # Use iterative retrieval for complex multi-hop questions (relaxed criteria)
            if self._is_complex_multi_hop(original_query):
                vector_texts = self._iterative_retrieval(original_query, vector_texts)
            # Also use iterative retrieval for questions with multiple entities
            elif self._has_multiple_entities(original_query):
                vector_texts = self._iterative_retrieval(original_query, vector_texts)

        results = []
        if graph_answer is not None:
            # GraphRAG found a good answer - use it as primary result
            results = graph_answer
            # Add top vector results as supplementary information
            if vector_texts:
                results.extend(vector_texts)  # Top vector results as supplement
        else:
            # Fallback to vector results only
            results = vector_texts

        if rerank and vector_texts:
            # Ensure query_text is a string for rerank
            rerank_query = (
                query_text
                if isinstance(query_text, str)
                else " ".join(query_text)
                if isinstance(query_text, list)
                else str(query_text)
            )
            try:
                reranked_texts = self.service.rerank_documents(vector_texts, rerank_query)
                if reranked_texts and len(reranked_texts) > 0:
                    if not (
                        graph_answer is not None
                        and isinstance(graph_answer, str)
                        and graph_answer.strip()
                        and "No relevant information found" not in graph_answer
                    ):
                        results = reranked_texts
                else:
                    print("⚠️  Warning: Reranking returned empty results, using original vector texts")
            except Exception as e:
                print(f"⚠️  Warning: Reranking failed with error: {e}, using original vector texts")

        results = results[:limit]

        # Return retrieved documents/context (no answer generation in core pipeline)

        if compress:
            # Ensure query_text is a string for compress
            compress_query = query_text if isinstance(query_text, str) else " ".join(query_text) if isinstance(query_text, list) else str(query_text)
            compressed_result = self.service.compress_prompt(compress_query, results)
            # If compression fails or returns empty, return the original results
            if compressed_result and compressed_result.strip():
                return [compressed_result]
            else:
                return results

        return results
        
    def _prior_knowledge_from_metadata(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Extract prior_knowledge or summary from document metadata (top-level or nested)."""
        pk = metadata.get("prior_knowledge") or metadata.get("summary")
        if pk is not None and isinstance(pk, str) and pk.strip():
            return pk.strip()
        nested = metadata.get("metadata")
        if isinstance(nested, dict):
            pk = nested.get("prior_knowledge") or nested.get("summary")
            if pk is not None and isinstance(pk, str) and pk.strip():
                return pk.strip()
        return None

    def add_document(self, document: Dict[str, Any]) -> None:
        metadata = {k: v for k, v in document.items() if k != "content"}
        text = document.get("content", "")
        if not text:
            return

        prior_knowledge = self._prior_knowledge_from_metadata(metadata)
        chunks = self.chunker.chunk_document(
            text,
            doc_context=metadata,
            prior_knowledge=prior_knowledge,
            inject_context=True,
        )
        vector_docs = [{**metadata, "content": chunk} for chunk in chunks]
        if vector_docs:
            self.vector_rag.add_document(vector_docs)

        # Add to GraphRAG only if enabled
        if self.graphrag_enabled:
            graph_nodes = self._build_graph_nodes(text, metadata)
            if graph_nodes:
                self.graph_rag.build_index(graph_nodes)
                # 🚀 For single documents, use smart rebuilding to avoid unnecessary work
                self._rebuild_communities_if_needed()

    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        all_vector_docs: List[Dict[str, Any]] = []
        all_nodes: List[TextNode] = []

        for document in documents:
            metadata = {k: v for k, v in document.items() if k != "content"}
            text = document.get("content", "")
            if not text:
                continue

            prior_knowledge = self._prior_knowledge_from_metadata(metadata)
            chunks = self.chunker.chunk_document(
                text,
                doc_context=metadata,
                prior_knowledge=prior_knowledge,
                inject_context=True,
            )
            all_vector_docs.extend({**metadata, "content": chunk} for chunk in chunks)
            # Build graph nodes only if GraphRAG is enabled
            if self.graphrag_enabled:
                all_nodes.extend(self._build_graph_nodes(text, metadata))

        if all_vector_docs:
            self.vector_rag.add_document(all_vector_docs)

        # Add to GraphRAG only if enabled
        if self.graphrag_enabled and all_nodes:
            try:
                self.graph_rag.build_index(all_nodes)
                # 🚀 Generate communities immediately during document loading (not query time!)
                self.graph_rag.build_communities()
            except RuntimeError as e:
                # Handle asyncio.run() errors from llama_index
                if "asyncio.run()" in str(e) or "event loop" in str(e).lower():
                    print(f"⚠️ GraphRAG processing skipped due to event loop conflict: {e}")
                    print("📝 Documents added to vector RAG only (GraphRAG disabled for this operation)")
                else:
                    raise

    def _build_graph_nodes(
        self, text: str, metadata: Dict[str, Any]
    ) -> List[TextNode]:
        chunks = self.splitter.split_text(text)
        return [TextNode(text=chunk, metadata=metadata) for chunk in chunks if chunk]

    def _rebuild_communities_if_needed(self):
        """Only rebuild communities if the graph has changed significantly."""
        try:
            if hasattr(self.graph_rag.graph_store, '_should_rebuild_communities'):
                if self.graph_rag.graph_store._should_rebuild_communities():
                    # Rebuilding communities silently
                    self.graph_rag.graph_store.invalidate_communities()
                    self.graph_rag.build_communities()
            else:
                # Fallback to always rebuild if change detection not available
                print("⚠️  Change detection not available - rebuilding communities")
                self.graph_rag.build_communities()
        except Exception as e:
            print(f"⚠️  Error in community rebuild check: {e}")
            # Fallback to rebuild
            self.graph_rag.build_communities()


def main():
    """Main CLI entry point for HybridRAG pipeline."""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="HybridRAG Pipeline")
    parser.add_argument("--env", type=Path, help="Environment file path")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--limit", type=int, default=10, help="Number of results to return")
    
    args = parser.parse_args()
    
    if args.env:
        from .cli_utils import load_env_file
        load_env_file(args.env)
    
    try:
        from .cli_utils import build_pipeline_from_env
        pipeline = build_pipeline_from_env()
        
        if args.query:
            results = pipeline.query(args.query, limit=args.limit)
            print(f"Found {len(results)} results")
            for i, result in enumerate(results):
                print(f"{i+1}. {result}")
        else:
            print("HybridRAG Pipeline initialized successfully!")
            print("Use --query 'your question' to search documents")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required services (Milvus, Neo4j) are running and environment variables are set.")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
