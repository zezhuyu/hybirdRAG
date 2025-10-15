import re
from typing import List

from llama_index.core import PropertyGraphIndex
from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.query_engine import CustomQueryEngine

from .store import GraphRAGStore


class GraphRAGQueryEngine(CustomQueryEngine):
    graph_store: GraphRAGStore
    index: PropertyGraphIndex
    llm: LLM
    similarity_top_k: int = 20

    def custom_query(self, query_str: str) -> str:
        """Process all community summaries to generate answers to a specific query."""
        # 🚀 ULTRA-FAST QUERY: Skip all expensive processing
        try:
            # Get all nodes quickly
            all_nodes = list(self.index.docstore.docs.values())
            
            # Filter nodes by query relevance using improved matching
            query_terms = query_str.lower().split()
            # Remove common stop words for better matching, but keep question words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can'}
            # Keep important question words and terms longer than 1 character
            query_terms = [term for term in query_terms if term not in stop_words and len(term) > 1]
            
            # If no terms remain after filtering, use the original query as a fallback
            if not query_terms:
                query_terms = [query_str.lower().strip()]
            
            relevant_nodes = []
            
            for node in all_nodes:
                if hasattr(node, 'text') and node.text:
                    node_text = node.text.lower()
                    
                    # Multiple matching strategies
                    match_score = 0
                    
                    # Strategy 1: Exact term matching
                    for term in query_terms:
                        if term in node_text:
                            match_score += 1
                    
                    # Strategy 2: Partial word matching (for compound terms)
                    for term in query_terms:
                        if any(term in word or word in term for word in node_text.split() if len(word) > 3):
                            match_score += 0.5
                    
                    # Strategy 3: Semantic similarity (simple keyword expansion)
                    semantic_groups = {
                        'person': ['person', 'people', 'individual', 'man', 'woman', 'human'],
                        'place': ['place', 'location', 'city', 'country', 'state', 'region'],
                        'time': ['time', 'year', 'date', 'period', 'era', 'century'],
                        'event': ['event', 'incident', 'happening', 'occurrence', 'situation']
                    }
                    
                    for term in query_terms:
                        for category, related_words in semantic_groups.items():
                            if term in related_words:
                                for related_word in related_words:
                                    if related_word in node_text:
                                        match_score += 0.3
                                        break
                    
                    # Accept nodes with any match
                    if match_score > 0:
                        relevant_nodes.append(node)
        
            if len(relevant_nodes) == 0:
                # Fallback: use first few nodes
                relevant_nodes = all_nodes[:5]
                print("⚠️  No relevant nodes found, using fallback")
            
            # Extract and clean text, prioritizing story content
            relevant_texts = []
            story_content = []
            metadata_content = []
            
            for node in relevant_nodes:
                if hasattr(node, 'text') and node.text:
                    clean_text = node.text.strip()
                    if len(clean_text) > 100:  # Only substantial text
                        # Clean up the text
                        clean_text = clean_text.replace('\n', ' ').replace('\r', ' ')
                        # Remove excessive whitespace
                        clean_text = ' '.join(clean_text.split())
                        
                        # Categorize content
                        if any(term in clean_text.lower() for term in [
                            "project gutenberg", "gutenberg", "etext", "donation", "copyright",
                            "foundation", "archive", "tax-deductible", "irs", "university ave",
                            "legal small print"
                        ]):
                            metadata_content.append(clean_text[:2000])  # Increased limit
                        else:
                            story_content.append(clean_text[:2000])  # Increased limit
            
            # Use story content for general queries
            if story_content:
                relevant_texts = story_content[:5]  # Use up to 5 story items
            else:
                relevant_texts = metadata_content[:3]  # Use up to 3 metadata items as fallback
            
            if relevant_texts:
                # Create a structured answer based on the query
                if "react" in query_str.lower() or "reaction" in query_str.lower():
                    # For reaction queries, provide a structured analysis
                    answer = f"Based on the text analysis, here's how the character reacts:\n\n"
                    for i, text in enumerate(relevant_texts[:5], 1):  # Increased from 3 to 5
                        answer += f"{i}. {text}\n\n"
                    return answer
                elif "what" in query_str.lower() or "how" in query_str.lower():
                    # For what/how queries, provide explanatory answers
                    answer = f"Based on the knowledge graph, here's what the text reveals:\n\n"
                    for i, text in enumerate(relevant_texts[:5], 1):  # Increased from 3 to 5
                        answer += f"{i}. {text}\n\n"
                    return answer
                else:
                    # General structured answer
                    answer = f"Based on the knowledge graph analysis:\n\n"
                    for i, text in enumerate(relevant_texts[:5], 1):  # Increased from 3 to 5
                        answer += f"{i}. {text}\n\n"
                    return answer
            else:
                return ""
                
        except Exception as e:
            print(f"⚠️  Ultra-fast retrieval failed: {e}")
            return "Unable to process query at this time."

    def get_entities(self, query_str: str, similarity_top_k: int) -> List[str]:
        # 🚀 FAST ENTITY EXTRACTION: Skip expensive processing
        # Quick entity extraction from query terms
        query_terms = [word.lower() for word in query_str.split() if len(word) > 2]
        entities = set(query_terms)
        
        # Add some common entity names from the text
        try:
            # Get a few nodes quickly
            all_nodes = list(self.index.docstore.docs.values())
            sample_nodes = all_nodes[:10]  # Only check first 10 nodes for speed
            
            for node in sample_nodes:
                if hasattr(node, 'text') and node.text:
                    # Extract capitalized words as potential entities
                    words = node.text.split()
                    for word in words:
                        if len(word) > 2 and word[0].isupper() and word.isalpha():
                            entities.add(word.lower())
            
            # Limit to reasonable number
            entities = list(entities)[:15]
            
        except Exception as e:
            print(f"⚠️  Fast extraction failed: {e}")
            # Fallback to query terms only
            entities = query_terms[:10]
        
        return entities

    def _is_quality_text(self, text: str) -> bool:
        """Check if text is of good quality for answering queries."""
        # Skip repetitive content
        words = text.split()
        if len(words) < 10:
            return False
            
        # Skip Project Gutenberg metadata
        if any(term in text.lower() for term in [
            'project gutenberg', 'gutenberg', 'etext', 'donation', 'copyright',
            'foundation', 'archive', 'tax-deductible', 'irs', 'university ave'
        ]):
            return False
            
        # Skip repetitive content
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Skip if any word appears more than 30% of the time (less strict)
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition > len(words) * 0.3:
            return False
            
        # Skip if text is mostly single repeated words (less strict)
        if len(set(words)) < len(words) * 0.3:
            return False
            
        # Skip very short sentences (less strict)
        sentences = text.split('.')
        if len(sentences) < 1:  # Allow single sentences
            return False
            
        # Skip text that's mostly numbers or special characters (less strict)
        alpha_chars = sum(1 for c in text if c.isalpha())
        if alpha_chars < len(text) * 0.5:  # Less strict
            return False
            
        return True

    def retrieve_entity_communities(self, entity_info, entities):
        """
        Retrieve cluster information for given entities, allowing for multiple clusters per entity.

        Args:
        entity_info (dict): Dictionary mapping entities to their cluster IDs (list).
        entities (list): List of entity names to retrieve information for.

        Returns:
        List of community or cluster IDs to which an entity belongs.
        """
        community_ids = []

        for entity in entities:
            if entity in entity_info:
                community_ids.extend(entity_info[entity])

        return list(set(community_ids))

    def generate_answer_from_summary(self, community_summary, query):
        """Generate an answer from a community summary based on a given query using LLM."""
        prompt = (
            f"Given the community summary: {community_summary}, "
            f"how would you answer the following query? Query: {query}"
        )
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content="I need an answer based on the above information.",
            ),
        ]
        response = self.llm.chat(messages)
        cleaned_response = re.sub(r"^assistant:\s*", "", str(response)).strip()
        return cleaned_response

    def aggregate_answers(self, community_answers):
        """Aggregate individual community answers into a final, coherent response."""
        # intermediate_text = " ".join(community_answers)
        prompt = "Combine the following intermediate answers into a final, concise response."
        messages = [
            ChatMessage(role="system", content=prompt),
            ChatMessage(
                role="user",
                content=f"Intermediate answers: {community_answers}",
            ),
        ]
        final_response = self.llm.chat(messages)
        cleaned_final_response = re.sub(
            r"^assistant:\s*", "", str(final_response)
        ).strip()
        return cleaned_final_response
