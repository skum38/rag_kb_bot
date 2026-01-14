"""
Knowledge Graph Utilities for RAG
Simplified KG extraction and querying
"""

import networkx as nx
import matplotlib.pyplot as plt
import pickle
import json
import re

from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from langchain_openai import ChatOpenAI

# Initialize LLM for extraction
kg_llm = None

def set_llm(llm):
    """Set the LLM to use for extraction"""
    global kg_llm
    kg_llm = llm


class SimpleKnowledgeGraph:
    """Simplified Knowledge Graph for RAG integration"""
    
    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_cache = {}
    
    def add_entity(self, name: str, entity_type: str, metadata: dict = None):
        """Add an entity to the graph"""
        if metadata is None:
            metadata = {}
        
        name = name.strip()
        if not self.graph.has_node(name):
            self.graph.add_node(name, type=entity_type, **metadata)
            self.entity_cache[name.lower()] = name
    
    def add_relationship(self, source: str, relation: str, target: str):
        """Add a relationship between entities"""
        source = source.strip()
        target = target.strip()
        relation = relation.upper().replace(" ", "_")
        
        # Ensure entities exist
        if not self.graph.has_node(source):
            self.add_entity(source, "UNKNOWN")
        if not self.graph.has_node(target):
            self.add_entity(target, "UNKNOWN")
        
        self.graph.add_edge(source, target, relation=relation)
    
    def get_entity_context(self, entity_name: str, max_depth: int = 2) -> str:
        """Get textual context about an entity from the graph"""
        entity_name = entity_name.strip()
        
        # Try case-insensitive lookup
        if entity_name not in self.graph:
            entity_name = self.entity_cache.get(entity_name.lower(), entity_name)
        
        if entity_name not in self.graph:
            return ""
        
        context_parts = []
        
        # Entity type
        entity_type = self.graph.nodes[entity_name].get('type', 'UNKNOWN')
        context_parts.append(f"**{entity_name}** ({entity_type})")
        
        # Outgoing relationships
        outgoing = []
        for _, target, data in self.graph.out_edges(entity_name, data=True):
            relation = data.get('relation', 'RELATED_TO')
            outgoing.append(f"  → {relation}: {target}")
        
        if outgoing:
            context_parts.append("Relationships:")
            context_parts.extend(outgoing[:5])  # Limit to 5
        
        # Incoming relationships
        incoming = []
        for source, _, data in self.graph.in_edges(entity_name, data=True):
            relation = data.get('relation', 'RELATED_TO')
            incoming.append(f"  ← {relation}: {source}")
        
        if incoming:
            context_parts.append("Related from:")
            context_parts.extend(incoming[:3])  # Limit to 3
        
        return "\n".join(context_parts)
    
    def find_path(self, start: str, end: str) -> List[Tuple[str, str, str]]:
        """Find path between two entities"""
        start = self.entity_cache.get(start.lower(), start)
        end = self.entity_cache.get(end.lower(), end)
        
        if start not in self.graph or end not in self.graph:
            return []
        
        try:
            path = nx.shortest_path(self.graph, start, end)
            path_relations = []
            
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    relation = list(edge_data.values())[0].get('relation', 'RELATED')
                    path_relations.append((path[i], relation, path[i+1]))
            
            return path_relations
        except:
            return []
    
    def search_entity(self, query: str) -> Optional[str]:
        """Find entity by partial name match"""
        query = query.lower()
        for entity_name in self.graph.nodes():
            if query in entity_name.lower():
                return entity_name
        return None
    
    def visualize(self, filename: str = "knowledge_graph.png"):
        """Create a simple visualization"""
        if self.graph.number_of_nodes() == 0:
            return
        
        plt.figure(figsize=(15, 10))
        pos = nx.spring_layout(self.graph, k=2, seed=42)
        
        # Color by type
        colors = []
        for node in self.graph.nodes():
            node_type = self.graph.nodes[node].get('type', 'UNKNOWN')
            if node_type == 'PERSON':
                colors.append('lightblue')
            elif node_type == 'CONCEPT':
                colors.append('lightgreen')
            elif node_type == 'DATE':
                colors.append('lightyellow')
            else:
                colors.append('lightgray')
        
        nx.draw_networkx_nodes(self.graph, pos, node_color=colors, node_size=2000, alpha=0.9)
        nx.draw_networkx_edges(self.graph, pos, edge_color='gray', arrows=True, alpha=0.6)
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight='bold')
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close()
    
    def save(self, filename: str = "kg.pkl"):
        """Save graph to disk"""
        with open(filename, 'wb') as f:
            pickle.dump({'graph': self.graph, 'cache': self.entity_cache}, f)
    
    def load(self, filename: str = "kg.pkl"):
        """Load graph from disk"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.graph = data['graph']
                self.entity_cache = data['cache']
        except FileNotFoundError:
            pass
    
    def get_stats(self) -> Dict:
        """Get graph statistics"""
        return {
            'entities': self.graph.number_of_nodes(),
            'relationships': self.graph.number_of_edges()
        }


def extract_entities_and_relations(text: str, llm) -> Dict:
    """
    Extract entities and relationships from text using LLM
    
    Returns: {"entities": [...], "relationships": [...]}
    """
    
    # Skip if text is too short
    if len(text.strip()) < 100:
        return {"entities": [], "relationships": []}
    
    prompt = f"""Extract entities and relationships from this text.
Focus on: people, concepts, dates, locations, and key ideas.

Text: {text[:2000]}

Return JSON only:
{{
    "entities": [{{"name": "Entity Name", "type": "PERSON|CONCEPT|DATE|LOCATION"}}],
    "relationships": [{{"source": "Entity1", "relation": "VERB", "target": "Entity2"}}]
}}

Keep it simple and factual. Extract 3-7 entities max."""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        # Clean JSON
        if "```json" in content:  
            content = content.split("```json")[1].split("```")[0]  
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        data = json.loads(content.strip())
        return data
    except Exception as e:
        return {"entities": [], "relationships": []}


def extract_question_entities(question: str, llm) -> List[str]:
    """Extract entity names mentioned in a question"""
    
    prompt = f"""What are the key entity names mentioned in this question?
Return only the names as a JSON list.

Question: {question}

Example output: ["Isaac Newton", "Gravity"]

Return JSON only: {{"entities": [...]}}"""

    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        
        if "```json" in content:  
            content = content.split("```json")[1].split("```")[0]  
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        data = json.loads(content.strip())
        return data.get("entities", [])
    except:
        return []


def extract_keywords_from_question(question: str) -> List[str]:
    """
    Extract important keywords from question without using LLM
    (faster for every query)
    """
    # Remove question words
    stop_words = {'what', 'is', 'are', 'who', 'where', 'when', 'how', 'why', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'and', 'or'}
    
    # Tokenize and clean
    words = re.findall(r'\b\w+\b', question.lower())
    keywords = [w for w in words if w not in stop_words and len(w) > 2]
    
    return keywords


def get_kg_context_enhanced(question: str, kg: SimpleKnowledgeGraph, llm) -> str:
    """
    Enhanced KG context retrieval - tries multiple strategies
    """
    if kg is None or kg.graph.number_of_nodes() == 0:
        return ""
    
    context_parts = []
    found_entities = []
    
    # Strategy 1: Quick keyword matching (fast)
    keywords = extract_keywords_from_question(question)
    for keyword in keywords:
        entity = kg.search_entity(keyword)
        if entity and entity not in found_entities:
            found_entities.append(entity)
    
    # Strategy 2: LLM extraction (if quick matching failed)
    if not found_entities:
        question_entities = extract_question_entities(question, llm)
        for entity_name in question_entities[:3]:
            entity = kg.search_entity(entity_name)
            if entity and entity not in found_entities:
                found_entities.append(entity)
    
    # Build context from found entities
    if found_entities:
        context_parts.append("=== KNOWLEDGE GRAPH CONTEXT ===\n")
        
        for entity in found_entities[:3]:  # Max 3 entities
            entity_context = kg.get_entity_context(entity)
            if entity_context:
                context_parts.append(entity_context)
                context_parts.append("")  # Blank line
        
        # Check paths between entities
        if len(found_entities) >= 2:
            path = kg.find_path(found_entities[0], found_entities[1])
            if path:
                context_parts.append("**Connection:**")
                for source, rel, target in path:
                    context_parts.append(f"  {source} → {rel} → {target}")
    
    return "\n".join(context_parts) if len(context_parts) > 1 else ""


def build_kg_from_chunks(chunks: List, llm, max_chunks: int = 30) -> SimpleKnowledgeGraph:
    """
    Build knowledge graph from document chunks
    
    Args:
        chunks: List of Document objects
        llm: Language model for extraction
        max_chunks: Maximum chunks to process (for speed)
    
    Returns:
        SimpleKnowledgeGraph
    """
    kg = SimpleKnowledgeGraph()
    
    # Process limited number of chunks
    for i, chunk in enumerate(chunks[:max_chunks]):
        # Extract from chunk
        kg_data = extract_entities_and_relations(chunk.page_content, llm)
        
        # Add entities
        for entity in kg_data.get("entities", []):
            kg.add_entity(
                entity["name"],
                entity["type"],
                {"source_page": chunk.metadata.get("page", 0)}
            )
        
        # Add relationships
        for rel in kg_data.get("relationships", []):
            kg.add_relationship(
                rel["source"],
                rel["relation"],
                rel["target"]
            )
    
    return kg