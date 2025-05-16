<Climb>
  <header>
    <id>mdc1</id>
    <type>feature</type>
    <description>Enhance Qdrant integration with advanced search and monitoring tools</description>
  </header>
  <newDependencies>None needed - will use existing Qdrant client and infrastructure</newDependencies>
  <prerequisitChanges>None - building on top of existing Qdrant integration</prerequisitChanges>
  <relevantFiles>
    - src/crawl4ai_mcp.py (main MCP server file)
    - src/utils.py (where helper functions are defined)
  </relevantFiles>

  <featureOverview>
    <featureName>Enhanced Qdrant Integration Tools</featureName>
    <purposeStatement>
      Extend the existing Qdrant integration with advanced search, monitoring, and recommendation capabilities to provide more powerful and flexible vector search operations.
    </purposeStatement>
    <problemSolved>
      The current integration with Qdrant provides basic vector search functionality but lacks advanced features like hybrid search, collection statistics, item-to-item recommendations, and vector clustering that would make the application more powerful and useful.
    </problemSolved>
    <successMetrics>
      - Successfully implement all four new tools
      - Maintain compatibility with existing functionality
      - Each tool should handle error conditions gracefully
      - Performance impact should be minimal
    </successMetrics>
  </featureOverview>

  <requirements>
    <functionalRequirements>
      1. Implement Hybrid Search Tool to combine vector similarity with keyword searching
      2. Create Collection Statistics Dashboard to monitor collection metrics
      3. Build Item-to-Item Recommendations tool for similar content discovery
      4. Develop Vector Clustering Tool to identify patterns in vector data
    </functionalRequirements>
    <technicalRequirements>
      - All tools must be implemented as MCP tools following the FastMCP pattern
      - Tools must be optimized to minimize memory and CPU usage
      - Tools should include comprehensive error handling
      - Documentation must be provided for each tool
    </technicalRequirements>
    <userRequirements>
      - Tools should have clear, consistent interfaces
      - Result formats should be JSON for easy parsing and display
      - Parameters should have sensible defaults
    </userRequirements>
    <constraints>
      - Must work with existing Qdrant client version
      - Should not require changes to collection structure
    </constraints>
  </requirements>

  <implementationDetails>
    <architectureOverview>
      Each tool will be implemented as an additional MCP tool in the crawl4ai_mcp.py file, using the existing Qdrant client from the context. Helper functions will be added to utils.py as needed.
    </architectureOverview>
    <toolDetails>
      <hybridSearchTool>
        <description>Combines vector similarity search with keyword/text-based filtering</description>
        <implementation>
          - Take query text and optional filter terms as input
          - Convert query to vector embedding
          - Perform vector search with keyword filtering
          - Return combined results
        </implementation>
        <parameters>
          - query: Text to search for
          - filter_text: Optional keyword filters
          - vector_weight: Weight for vector results (0.0-1.0)
          - keyword_weight: Weight for keyword results (0.0-1.0)
          - match_count: Number of results to return
        </parameters>
      </hybridSearchTool>
      
      <collectionStatsTool>
        <description>Monitors collection size, point count, and memory usage</description>
        <implementation>
          - Query Qdrant collection info endpoints
          - Process and aggregate statistics
          - Format results as structured data
        </implementation>
        <parameters>
          - collection_name: Optional collection to analyze (defaults to all)
          - include_segments: Whether to include segment-level details
        </parameters>
      </collectionStatsTool>
      
      <itemRecommendationTool>
        <description>Given an item ID, finds similar items based on vector similarity</description>
        <implementation>
          - Take item ID as input
          - Use Qdrant recommendation API to find similar items
          - Apply optional filtering
          - Return results with similarity scores
        </implementation>
        <parameters>
          - item_id: ID of the item to find recommendations for
          - filter: Optional filter criteria
          - match_count: Number of recommendations to return
        </parameters>
      </itemRecommendationTool>
      
      <vectorClusteringTool>
        <description>Groups similar vectors to identify data patterns</description>
        <implementation>
          - Query vectors from Qdrant
          - Apply clustering algorithm (K-means initially)
          - Return cluster assignments and statistics
        </implementation>
        <parameters>
          - collection_name: Collection to analyze
          - num_clusters: Number of clusters to create
          - random_seed: Seed for reproducibility
          - sample_size: Maximum vectors to process
        </parameters>
      </vectorClusteringTool>
    </toolDetails>
  </implementationDetails>

  <testingApproach>
    <testCases>
      - Test each tool with valid parameters
      - Test error handling with invalid inputs
      - Test performance with various dataset sizes
    </testCases>
    <acceptanceCriteria>
      - All tools return expected results format
      - Error messages are clear and helpful
      - Performance is within acceptable limits
    </acceptanceCriteria>
  </testingApproach>
</Climb> 