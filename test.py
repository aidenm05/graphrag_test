import os
import asyncio
from dotenv import load_dotenv
import neo4j
from neo4j_graphrag.llm import OpenAILLM
from neo4j_graphrag.embeddings.openai import OpenAIEmbeddings
from neo4j_graphrag.experimental.components.text_splitters.fixed_size_splitter import FixedSizeSplitter
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline

# Load environment variables
load_dotenv()

# Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize Neo4j driver
driver = neo4j.GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
)


# Define schema components
basic_node_labels = [
    "Object", "Entity", "Person", "Group", 
    "Organization", "Place", "Event"
]
interpersonal_node_labels = [
    "Conversation", "Relationship", "Role",
    "Gesture", "Emotion"
]
specialized_node_labels = [
    "Utterance", "Paralinguistic", "Topic",
    "Turn", "Repair"
]
relationships = [
    "PARTICIPATES_IN", "SPOKEN_BY", "CONTAINS",
    "MENTIONS", "EXPRESSES", "OCCURS_AT", "HAS_ROLE",
    "RESPONDS_TO", "INTERRUPTS", "REFERENCES",
    "TRIGGERS", "PRECEDES", "REPAIRS"
]

# Format prompt template with actual values
# ... [previous imports and setup code remains the same]
prompt_template = '''
You are a sociolinguist analyzing conversational transcripts from the UCSB Santa Barbara Corpus of Spoken American English (TRM files). 
Extract entities and relationships to model interpersonal dynamics using the following schema:

Extract the entities (nodes) and specify their type from the following text.
Also extract the relationships between these nodes.

### Node Types:
- Basic: {basic_node_labels}
- Interpersonal: {interpersonal_node_labels}
- Specialized: {specialized_node_labels}

### Relationship Types: {relationships}

### Instructions:
1. **Entity Extraction**:
   - Identify all speakers (`Person`), groups (`Group`), and mentioned entities
   - Tag paralinguistic features (`Paralinguistic`)
   - Capture emotional states (`Emotion`) and social roles (`Role`)

2. **Relationship Mapping**:
   - Connect utterances to speakers using `SPOKEN_BY`
   - Map conversation flow with `PRECEDES` and `RESPONDS_TO`
   - Link emotional expressions with `EXPRESSES`
   - Track interruptions with `INTERRUPTS`

3. **TRM-Specific Features**:
   - Preserve timestamps using `start_time`/`end_time`
   - Mark overlapping speech with `is_overlap: true`
   - Include repair strategies (`Repair` nodes)

Return the result as JSON using the following format:
{{"nodes": [ {{"id": "0", "label": "entity type", "properties": {{"name": "entity name"}} }} ],
  "relationships": [{{"type": "RELATIONSHIP_TYPE", "start_node_id": "0", "end_node_id": "1", "properties": {{"details": "Relationship details"}} }}] }}

Input text:

{text}
'''

# # Define the prompt template with properly escaped curly braces
# prompt_template = f'''
# You are a sociolinguist analyzing conversational transcripts from the UCSB Santa Barbara Corpus of Spoken American English (TRM files). 
# Extract entities and relationships to model interpersonal dynamics using the following schema:

# ### Node Types:
# - Basic: {basic_node_labels}
# - Interpersonal: {interpersonal_node_labels}
# - Specialized: {specialized_node_labels}

# ### Relationship Types: {relationships}

# ### Instructions:
# 1. **Entity Extraction**:
#    - Identify all speakers (`Person`), groups (`Group`), and mentioned entities
#    - Tag paralinguistic features (`Paralinguistic`)
#    - Capture emotional states (`Emotion`) and social roles (`Role`)

# 2. **Relationship Mapping**:
#    - Connect utterances to speakers using `SPOKEN_BY`
#    - Map conversation flow with `PRECEDES` and `RESPONDS_TO`
#    - Link emotional expressions with `EXPRESSES`
#    - Track interruptions with `INTERRUPTS`

# 3. **TRM-Specific Features**:
#    - Preserve timestamps using `start_time`/`end_time`
#    - Mark overlapping speech with `is_overlap: true`
#    - Include repair strategies (`Repair` nodes)

# ### Output Format (JSON):
# {{
#   "nodes": [
#     {{{{ "id": "unique_id", "label": "NodeType", "properties": {{{{ "key": "value" }}}} }}}},
#   ],
#   "relationships": [
#     {{{{ "type": "RELATIONSHIP_TYPE", "start_node_id": "X", "end_node_id": "Y" }}}}
#   ]
# }}

# Example Input:
# "(0.4) Alice: So um [laughs] you really think that? (.) I mean-"

# Example Output:
# {{
#   "nodes": [
#     {{{{ "id": "0", "label": "Person", "properties": {{{{ "name": "Alice" }}}} }}}},
#     {{{{ "id": "1", "label": "Paralinguistic", "properties": {{{{ "type": "laughter", "timestamp": 0.4 }}}} }}}},
#     {{{{ "id": "2", "label": "Utterance", "properties": {{{{ "text": "So um...", "start_time": 0.4, "is_overlap": false }}}} }}}}
#   ],
#   "relationships": [
#     {{{{ "type": "SPOKEN_BY", "start_node_id": "2", "end_node_id": "0" }}}},
#     {{{{ "type": "CONTAINS", "start_node_id": "2", "end_node_id": "1" }}}}
#   ]
# }}

# Input Text:
# {{{{
#     "text": "{text}"
# }}}}
# '''


# Initialize LLM and embeddings
llm = OpenAILLM(
    model_name="gpt-4o-mini",
    model_params={"response_format": {"type": "json_object"}, "temperature": 0}
)
embedder = OpenAIEmbeddings()

# Create KG pipeline
kg_builder_pdf = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    text_splitter=FixedSizeSplitter(chunk_size=1000, chunk_overlap=200),
    embedder=embedder,
    entities=basic_node_labels + interpersonal_node_labels + specialized_node_labels,
    relations=relationships,
    prompt_template=prompt_template,
    from_pdf=True
)


pdf_file_paths = [
    'C:\\Users\\aiden\\Downloads\\neo4j_experiements\\TRNPDF\\SBC049.pdf',
    'C:\\Users\\aiden\\Downloads\\neo4j_experiements\\TRNPDF\\SBC050.pdf'
]

async def main():
    for path in pdf_file_paths:
        print(f"Processing: {path}")
        pdf_result = await kg_builder_pdf.run_async(file_path=path)
        print(f"Result: {pdf_result}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())