# Reciprocal Rank Fusion (RRF) for Source-Aware Search

## Overview

Reciprocal Rank Fusion (RRF) is a rank aggregation algorithm that combines rankings from multiple search methods into a unified result set. This document outlines our research findings and implementation plan for using RRF to improve source filtering in our RAG system.

## The Problem We're Solving

Our vector database contains hundreds of sources (documentation sites, GitHub repos, etc.). Users shouldn't need to memorize exact domain names. When a user searches for "cursor features" with source "cursor", they should get:

1. **Prioritized results** from cursor-related sources (`docs.cursor.com`, `api.cursor.com`, etc.)
2. **Fallback results** from other sources if they're highly relevant
3. **Intuitive behavior** without requiring exact domain matching

## RRF Algorithm

### Formula
```
RRF(d) = Σ(r ∈ R) 1 / (k + r(d))
```

Where:
- `d` = document
- `R` = set of rankers (retrievers)
- `k` = constant (typically 60)
- `r(d)` = rank of document d in ranker r

### Mathematical Intuition

1. **Reciprocal Ranking**: Uses `1/(rank + k)` to give more weight to higher ranks
2. **Diminishing Returns**: Contribution decreases non-linearly as rank increases
3. **Rank Aggregation**: Sums reciprocal ranks across all retrievers for robustness
4. **Normalization**: Constant `k` acts as smoothing factor and prevents domination

### The k=60 Standard

- **Empirical Performance**: Studies show k=60 performs well across datasets
- **Balancing Influence**: Good balance between top-ranked and lower-ranked items
- **Effective Tie-Breaking**: Helps break ties, especially for lower-ranked items
- **Robustness**: Proven robust across different retrieval systems

## Qdrant Implementation

### Native RRF Support

Qdrant v1.10+ has built-in RRF support via the Query API:

```python
from qdrant_client import QdrantClient, models

client.query_points(
    collection_name="collection_name",
    prefetch=[
        models.Prefetch(
            query=models.SparseVector(indices=[1, 42], values=[0.22, 0.8]),
            using="sparse",
            limit=20,
        ),
        models.Prefetch(
            query=[0.01, 0.45, 0.67],  # dense vector
            using="dense",
            limit=20,
        ),
    ],
    query=models.FusionQuery(fusion=models.Fusion.RRF),
)
```

### Query API Pattern

1. **Prefetch**: Perform multiple sub-queries
2. **Fusion**: Apply RRF to combine results
3. **Server-side**: All processing happens on Qdrant server

## Our Implementation Plan

### Step 1: Source Discovery
```python
def discover_matching_sources(user_source: str, all_sources: List[str]) -> List[str]:
    """Find all sources that match the user's fuzzy input"""
    user_lower = user_source.lower()
    return [
        source for source in all_sources 
        if user_lower in source.lower() or source.lower() in user_lower
    ]
```

### Step 2: Dual Query Strategy
```python
async def perform_source_aware_search(query: str, source_hint: str, limit: int):
    # Query 1: Filtered search (source-specific)
    filtered_query = models.Prefetch(
        query=await get_embedding(query),
        query_filter=create_source_filter(matching_sources),
        limit=limit * 2
    )
    
    # Query 2: Global search (no filter)
    global_query = models.Prefetch(
        query=await get_embedding(query),
        limit=limit * 2
    )
    
    # Combine with RRF
    return await client.query_points(
        collection_name=collection_name,
        prefetch=[filtered_query, global_query],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit
    )
```

### Step 3: Implementation Architecture

```
User Query: "cursor features" + source: "cursor"
     ↓
Step 1: Discover Sources
     ├─ all_sources = get_available_sources()
     ├─ matching = ["docs.cursor.com", "api.cursor.com"]
     ↓
Step 2: Dual Prefetch
     ├─ Query A: semantic search + source filter
     ├─ Query B: semantic search (no filter)
     ↓
Step 3: RRF Fusion (server-side)
     ├─ Documents in both queries get boosted
     ├─ Pure semantic matches still included
     ↓
Result: Prioritized, relevant results
```

## Benefits of This Approach

### 1. **Friction Reduction**
- Users can type partial source names
- Fuzzy matching finds related sources
- No need to memorize exact domains

### 2. **Balanced Results**
- Source-matching documents get priority
- Highly relevant off-source documents still included
- RRF prevents complete filtering

### 3. **Performance**
- Server-side processing in Qdrant
- Native RRF implementation
- Parallel query execution

### 4. **Maintainability**
- Standard algorithm with proven effectiveness
- Clean separation of concerns
- Extensible to additional ranking signals

## Alternative Approaches Considered

### 1. **Linear Score Combination** ❌
```python
final_score = 0.7 * semantic_score + 0.3 * source_bonus
```
**Rejected**: Research shows this doesn't work well because scores aren't linearly separable.

### 2. **Client-side Score Manipulation** ❌
```python
for result in results:
    if matches_source(result):
        result.score += bonus
```
**Rejected**: Complex, error-prone, and doesn't leverage Qdrant's optimizations.

### 3. **Hard Filtering** ❌
```python
results = search_with_filter(source_filter)
if not results:
    results = search_without_filter()
```
**Rejected**: Too rigid, doesn't blend results effectively.

## Implementation Checklist

- [ ] Create `discover_matching_sources()` utility function
- [ ] Modify `perform_rag_query()` to use dual prefetch pattern
- [ ] Add RRF fusion via Qdrant Query API
- [ ] Update error handling for new query structure
- [ ] Add debug information to show RRF scoring
- [ ] Test with various source patterns
- [ ] Document new behavior for users

## References

- [Original RRF Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- [Qdrant Hybrid Queries Documentation](https://qdrant.tech/documentation/concepts/hybrid-queries/)
- [RRF Mathematical Intuition](https://medium.com/@devalshah1619/mathematical-intuition-behind-reciprocal-rank-fusion-rrf-explained-in-2-mins-002df0cc5e2a)
- [Azure AI Search RRF Implementation](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)
- [Qdrant Filtering Guide](https://medium.com/@vandriichuk/comprehensive-guide-to-filtering-in-qdrant-9fa5e9ad8e7b)

## Next Steps

1. **Implement** the dual prefetch pattern in `perform_rag_query()`
2. **Test** with real queries to validate effectiveness
3. **Optimize** the source discovery algorithm
4. **Monitor** performance impact of dual queries
5. **Iterate** based on user feedback 

# Reciprocal Rank Fusion (RRF) Implementation Plan

## Research Summary

RRF is the "de facto standard" for combining multiple search rankings. The formula is:
```
RRF(d) = Σ(r ∈ R) 1 / (k + r(d))
```
where k=60 is the standard parameter.

## Server-side RRF Requirements

### **Current Collection Status** ❌
- **Sparse Vectors**: `"sparse_vectors": "None"`
- **Only Dense Vectors**: Single unnamed vector (1536-dim embeddings)
- **No Text/BM25 Index**: Required for keyword search component

### **Requirements for Server-side RRF** ✅

1. **Sparse Vector Support**
   ```python
   sparse_vectors_config = {
       "text-sparse": models.SparseVectorParams(
           index=models.SparseIndexParams(on_disk=False)
       )
   }
   ```

2. **Collection Recreation with Both Vector Types**
   ```python
   client.create_collection(
       collection_name="crawl4ai_mcp",
       vectors_config={
           "text-dense": models.VectorParams(
               size=1536,
               distance=models.Distance.COSINE
           )
       },
       sparse_vectors_config={
           "text-sparse": models.SparseVectorParams()
       }
   )
   ```

3. **Sparse Vector Generation**
   - **SPLADE Models**: `naver/efficient-splade-VI-BT-large-doc` (documents) and `naver/efficient-splade-VI-BT-large-query` (queries)
   - **Alternative**: TF-IDF/BM25 sparse vectors
   - **FastEmbed**: Built-in sparse vector support

4. **Query API Structure**
   ```python
   client.query_points(
       collection_name="crawl4ai_mcp",
       prefetch=[
           models.Prefetch(
               query=dense_vector,
               using="text-dense",
               limit=match_count * 2
           ),
           models.Prefetch(
               query=models.SparseVector(indices=indices, values=values),
               using="text-sparse", 
               limit=match_count * 2
           )
       ],
       query=models.FusionQuery(fusion=models.Fusion.RRF),
       limit=match_count
   )
   ```

## Implementation Options

### **Option 1: Collection Recreation** (Recommended)
- **Pros**: Full server-side RRF support, optimal performance
- **Cons**: Requires re-indexing all data with sparse vectors
- **Effort**: High (need to generate sparse vectors for 1.6M+ points)

### **Option 2: Client-side RRF** (Current Alternative)
- **Pros**: Works with existing collection, no data migration
- **Cons**: Less efficient, more complex code
- **Effort**: Medium (already partially implemented)

### **Option 3: Hybrid Approach**
- **Pros**: Gradual migration, immediate benefits
- **Cons**: Temporary complexity
- **Effort**: Medium

## Performance Benefits

**Server-side RRF advantages:**
- **10x+ faster**: No client-side result combination
- **Better ranking**: Native RRF fusion algorithm
- **Scalability**: Handles large result sets efficiently
- **Simplicity**: Single query call vs multiple + fusion

## Next Steps

1. **Evaluate**: Determine if server-side RRF justifies collection recreation
2. **Sparse Model**: Choose SPLADE vs TF-IDF for sparse vectors
3. **Migration**: Plan data re-indexing strategy
4. **Testing**: Compare performance with current approach

## Resources

- [Qdrant Sparse Vectors Guide](https://qdrant.tech/articles/sparse-vectors/)
- [Hybrid Search with Query API](https://qdrant.tech/articles/hybrid-search/)
- [SPLADE Implementation Guide](https://viraajkadam.medium.com/a-guide-to-hybrid-search-using-splade-qdrant-vector-database-a4b70e243f4a) 