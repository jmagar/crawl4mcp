import os
import json
import asyncio
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# --- Early .env loading and variable setup ---
# This section should be as close to the top as possible.

# Attempt to load .env file first.
_dotenv_loaded = False
try:
    from dotenv import load_dotenv
    # Try to find .env in project root (assuming scripts/ is one level down)
    _dotenv_path = os.path.join(os.path.dirname(__file__), '..', '.env')
    if os.path.exists(_dotenv_path):
        load_dotenv(dotenv_path=_dotenv_path)
        print(f"INFO: Loaded .env file from {_dotenv_path} at script startup.")
        _dotenv_loaded = True
    else:
        # Fallback for cases where script might be run from project root directly
        _dotenv_path_alt = os.path.join(os.getcwd(), '.env')
        if os.path.exists(_dotenv_path_alt):
            load_dotenv(dotenv_path=_dotenv_path_alt)
            print(f"INFO: Loaded .env file from {_dotenv_path_alt} at script startup.")
            _dotenv_loaded = True
        else:
            print("WARNING: No .env file found at common locations at script startup.")
except ImportError:
    print("WARNING: python-dotenv not found. Cannot load .env file. Script will rely on system environment variables or defaults.")

# Now define constants, attempting to use environment variables
# These will use system env vars if .env loading failed or vars are not in .env

CONVERSATIONS_FILE_PATH = os.getenv("CONVERSATIONS_FILE_PATH", "conversations.json")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") # Can be None
QDRANT_COLLECTION_NAME = os.getenv("CLAUDE_HISTORY_COLLECTION", "claude_conversation_history")

# Crucial: Get EMBEDDING_SERVER_BATCH_SIZE from env AFTER potential .env load
# Default to a very small number if not set, to avoid Payload Too Large.
_default_embedding_batch_size = 8 
EMBEDDING_BATCH_SIZE_FROM_ENV = os.getenv("EMBEDDING_SERVER_BATCH_SIZE")
print(f"DEBUG: Value of EMBEDDING_SERVER_BATCH_SIZE from environment: {EMBEDDING_BATCH_SIZE_FROM_ENV}")

if EMBEDDING_BATCH_SIZE_FROM_ENV is not None:
    try:
        EMBEDDING_BATCH_SIZE = int(EMBEDDING_BATCH_SIZE_FROM_ENV)
        print(f"INFO: Using EMBEDDING_BATCH_SIZE from environment: {EMBEDDING_BATCH_SIZE}")
    except ValueError:
        print(f"WARNING: EMBEDDING_SERVER_BATCH_SIZE ('{EMBEDDING_BATCH_SIZE_FROM_ENV}') is not a valid integer. Using default: {_default_embedding_batch_size}")
        EMBEDDING_BATCH_SIZE = _default_embedding_batch_size
else:
    print(f"INFO: EMBEDDING_SERVER_BATCH_SIZE not found in environment. Using default: {_default_embedding_batch_size}")
    EMBEDDING_BATCH_SIZE = _default_embedding_batch_size

PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", "100")) # How many messages to process before Qdrant upsert
QDRANT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "128")) # New: Batch size for Qdrant client upsert calls
MAX_METADATA_STRING_LENGTH = int(os.getenv("MAX_METADATA_STRING_LENGTH", "2048")) # Max length for tool I/O strings in metadata

# Placeholder for project-specific imports if they fail
_project_modules_available = False
try:
    from src.utils.logging_utils import get_logger
    from src.utils.embedding_utils import create_embeddings_batch, VECTOR_DIM as PROJECT_VECTOR_DIM, EMBEDDING_SERVER_URL as PROJECT_EMBEDDING_SERVER_URL
    from src.utils.qdrant.setup import get_qdrant_client, ensure_qdrant_collection_async
    from src.utils.crawling_utils import smart_chunk_markdown, MARKDOWN_CHUNK_SIZE as PROJECT_MARKDOWN_CHUNK_SIZE
    from qdrant_client.http.models import PointStruct
    _project_modules_available = True
    SCRIPT_MARKDOWN_CHUNK_SIZE = PROJECT_MARKDOWN_CHUNK_SIZE # Use project's
    print("INFO: Successfully imported project-specific modules (src.utils.*).")
except ImportError as e:
    print(f"WARNING: Error importing project modules: {e}. Script will use placeholder functions for core logic.")
    SCRIPT_MARKDOWN_CHUNK_SIZE = 500 # Fallback for this script
    # Define placeholders if imports fail
    def get_logger(name):
        import logging
        logger = logging.getLogger(name)
        if not logger.handlers: # Avoid duplicate handlers
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        return logger
    VECTOR_DIM = 1024 # Fallback
    EMBEDDING_SERVER_URL = "http://localhost:7862/embed" # Fallback
    
    async def create_embeddings_batch(texts: List[str]) -> List[Optional[List[float]]]: 
        print("WARNING: Using placeholder create_embeddings_batch.")
        return [[0.0]*VECTOR_DIM for _ in texts] # Return dummy embeddings
    
    def get_qdrant_client():
        print("WARNING: Using placeholder get_qdrant_client. Will use global QDRANT_URL and QDRANT_API_KEY.")
        class MockQdrantClient:
            def __init__(self, url, api_key=None): 
                self.url = url; self.api_key = api_key
                print(f"MockQdrantClient initialized with URL: {self.url}")
            async def ensure_collection(self, name, dim):
                print(f"Mock ensure_collection for {name} with dim {dim}"); await asyncio.sleep(0)
            async def upsert(self, collection_name, points): 
                print(f"Mock upsert to {collection_name} with {len(points)} points"); await asyncio.sleep(0)
            async def get_collection(self, collection_name):
                 print(f"Mock get_collection for {collection_name}"); await asyncio.sleep(0)
                 class MockCollectionInfo: pass
                 return MockCollectionInfo()
        return MockQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    async def ensure_qdrant_collection_async(client, name, dim): 
        print(f"Mock ensure_qdrant_collection_async for {name} using client: {type(client)}"); 
        if hasattr(client, 'ensure_collection'): await client.ensure_collection(name, dim)
        else: print(f"Fallback mock ensure_qdrant_collection_async for {name} with dim {dim}"); 
        await asyncio.sleep(0)

    def smart_chunk_markdown(text: str, chunk_size: Optional[int] = None) -> List[str]:
        print(f"WARNING: Using placeholder smart_chunk_markdown. Target chunk_size: {chunk_size or SCRIPT_MARKDOWN_CHUNK_SIZE}")
        size = chunk_size or SCRIPT_MARKDOWN_CHUNK_SIZE
        if len(text) <= size: return [text]
        return [text[i:i+size] for i in range(0, len(text), size)]

    PointStruct = dict # Placeholder

# Use project variables if available, otherwise fall back to script-defined ones (which also check os.getenv)
EMBEDDING_SERVER_URL = PROJECT_EMBEDDING_SERVER_URL if _project_modules_available else os.getenv("EMBEDDING_SERVER_URL", "http://localhost:8001/v1/embeddings")
VECTOR_DIM = PROJECT_VECTOR_DIM if _project_modules_available else int(os.getenv("VECTOR_DIM", "384"))

# Initialize logger after constants are set up
logger = get_logger(__name__)
logger.info(f"Using SCRIPT_MARKDOWN_CHUNK_SIZE: {SCRIPT_MARKDOWN_CHUNK_SIZE}") # Log the chunk size being used
logger.info(f"Using MAX_METADATA_STRING_LENGTH: {MAX_METADATA_STRING_LENGTH}") # Log truncation length

if not _dotenv_loaded and not EMBEDDING_BATCH_SIZE_FROM_ENV: # Be more explicit if .env failed AND no system env for batch size
    logger.warning("EMBEDDING_BATCH_SIZE is using a script default because .env file was not loaded/found AND EMBEDDING_SERVER_BATCH_SIZE was not set as a system environment variable.")

# Removed MAX_TEXT_SEGMENT_LENGTH as we are now chunking properly

# --- Helper Functions ---

def stringify_content_item(item: Any) -> str:
    """Converts a content item (potentially complex dict/list) to a string."""
    if isinstance(item, str):
        return item
    if isinstance(item, (list, dict)):
        try:
            return json.dumps(item, indent=2) # Pretty print for better readability if it's structured
        except TypeError:
            return str(item) # Fallback for non-serializable objects
    return str(item)

def process_conversation_for_embedding(conversation: Dict[str, Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extracts and chunks relevant text segments and metadata from a single Claude conversation object.
    Each chunk will be an item to embed.
    """
    conversation_uuid = conversation.get("uuid", str(uuid.uuid4()))
    conversation_name = conversation.get("name", "Untitled Conversation")
    chat_messages = conversation.get("chat_messages", [])
    created_at = conversation.get("created_at", datetime.utcnow().isoformat())
    updated_at = conversation.get("updated_at", created_at)

    segments_data: List[Tuple[str, Dict[str, Any]]] = [] # Stores (text_to_embed, payload_template)

    for msg_idx, message in enumerate(chat_messages):
        message_uuid = message.get("uuid", str(uuid.uuid4()))
        speaker = message.get("speaker", "unknown")
        text_content_parts = message.get("content", [])
        message_created_at = message.get("created_at", created_at)

        base_payload_template = {
            "conversation_uuid": conversation_uuid,
            "conversation_name": conversation_name,
            "message_uuid": message_uuid,
            "message_index": msg_idx,
            "speaker": speaker,
            "message_created_at": message_created_at,
            "conversation_created_at": created_at,
            "conversation_updated_at": updated_at,
            "source_system": "claude_export"
            # "text_to_embed" and "sub_chunk_index" will be added per chunk
        }

        for content_idx, content_part in enumerate(text_content_parts):
            text_segment = ""
            content_part_type = "unknown"
            additional_payload = {} # For tool-specific fields

            if isinstance(content_part, str):
                content_part_type = "text"
                text_segment = content_part
            elif isinstance(content_part, dict):
                content_part_type = content_part.get("type", "dict_unknown")
                if content_part_type == "text":
                    text_segment = content_part.get("text", "")
                elif content_part_type == "tool_use":
                    tool_name = content_part.get("name", "unknown_tool")
                    tool_input_raw = content_part.get("input", {})
                    tool_input_str = stringify_content_item(tool_input_raw)
                    text_segment = f"Tool use: {tool_name}\nInput: {tool_input_str}"
                    additional_payload["tool_name"] = tool_name
                    
                    # Prepare tool_input_json for metadata payload, with truncation
                    tool_input_json_for_payload = json.dumps(tool_input_raw) if isinstance(tool_input_raw, dict) else tool_input_str
                    if len(tool_input_json_for_payload) > MAX_METADATA_STRING_LENGTH:
                        tool_input_json_for_payload = tool_input_json_for_payload[:MAX_METADATA_STRING_LENGTH] + "...[TRUNCATED]"
                        # logger.warning(f"Truncated tool_input_json in metadata for tool '{tool_name}'. Original length: {len(tool_input_str)}") # Can be noisy
                    additional_payload["tool_input_json"] = tool_input_json_for_payload
                    additional_payload["tool_id"] = content_part.get("id") # Claude tool_use can have an ID
                elif content_part_type == "tool_result":
                    tool_name = content_part.get("tool_name", "unknown_tool_from_result") # Claude's format might use 'tool_name'
                    tool_id = content_part.get("tool_use_id") # Common field for linking result to use
                    is_error = content_part.get("is_error", False)
                    result_data = content_part.get("content", "") # 'content' usually holds the actual result string or structured data

                    # Attempt to get text from various possible structures of result_data
                    if isinstance(result_data, list) and result_data and isinstance(result_data[0], dict) and "text" in result_data[0]:
                        actual_result_text = result_data[0].get("text", stringify_content_item(result_data))
                    elif isinstance(result_data, str):
                        actual_result_text = result_data
                    else:
                        actual_result_text = stringify_content_item(result_data)
                    
                    status_prefix = "Tool Error" if is_error else "Tool Result"
                    text_segment = f"{status_prefix} from {tool_name} (ID: {tool_id}):\n{actual_result_text}"
                    additional_payload["tool_name"] = tool_name
                    additional_payload["tool_use_id"] = tool_id
                    additional_payload["tool_is_error"] = is_error
                    
                    # Prepare tool_result_json for metadata payload, with truncation
                    tool_result_json_for_payload = json.dumps(result_data) if not isinstance(result_data, str) else actual_result_text # If it was already string, use actual_result_text
                    if len(tool_result_json_for_payload) > MAX_METADATA_STRING_LENGTH:
                        original_len = len(json.dumps(result_data) if not isinstance(result_data, str) else actual_result_text)
                        tool_result_json_for_payload = tool_result_json_for_payload[:MAX_METADATA_STRING_LENGTH] + "...[TRUNCATED]"
                        # logger.warning(f"Truncated tool_result_json in metadata for tool '{tool_name}'. Original length: {original_len}") # Can be noisy
                    additional_payload["tool_result_json"] = tool_result_json_for_payload

                else: # Other dict types
                    text_segment = stringify_content_item(content_part)
            else: # Other data types
                text_segment = stringify_content_item(content_part)

            if not text_segment or not text_segment.strip():
                logger.debug(f"Empty text_segment for msg {msg_idx}, content part {content_idx}. Skipping.")
                continue

            # --- Chunk the text_segment ---
            # Using MARKDOWN_CHUNK_SIZE (default 750 from crawling_utils or script fallback)
            # This size is for characters.
            sub_chunks = smart_chunk_markdown(text_segment, chunk_size=SCRIPT_MARKDOWN_CHUNK_SIZE)
            
            for sub_chunk_idx, chunk_text in enumerate(sub_chunks):
                if not chunk_text or not chunk_text.strip():
                    logger.debug(f"Empty sub_chunk for msg {msg_idx}, content part {content_idx}, sub_chunk {sub_chunk_idx}. Skipping.")
                    continue

                chunk_payload = base_payload_template.copy()
                chunk_payload.update(additional_payload) # Add tool-specific fields
                chunk_payload["content_part_index"] = content_idx
                chunk_payload["content_part_type"] = content_part_type
                chunk_payload["sub_chunk_index"] = sub_chunk_idx
                # The actual text for embedding is chunk_text, which will be handled by main loop

                segments_data.append((chunk_text, chunk_payload))
                
                # Removed truncation logic based on MAX_TEXT_SEGMENT_LENGTH
                # The chunking itself handles the size.

    return segments_data

async def main():
    logger.info("DEBUG: main() called.") # New log
    logger.info("Starting Claude conversation history ingestion script.")
    logger.info(f"Attempting to load conversations from: {CONVERSATIONS_FILE_PATH}")
    logger.info(f"Qdrant URL: {QDRANT_URL}")
    logger.info(f"Qdrant Collection: {QDRANT_COLLECTION_NAME}")
    logger.info(f"Vector Dimension: {VECTOR_DIM}")
    logger.info(f"Using Embedding Server URL: {EMBEDDING_SERVER_URL}") # From embedding_utils or script default
    logger.info(f"Using Embedding Server Batch Size (for HTTP calls by create_embeddings_batch): {EMBEDDING_BATCH_SIZE}")
    logger.info(f"Using Processing Batch Size (script's loop for Qdrant upserts): {PROCESSING_BATCH_SIZE}")
    logger.info(f"Using Qdrant Upsert Batch Size (points per Qdrant upsert call): {QDRANT_UPSERT_BATCH_SIZE}")
    logger.info(f"Max metadata string length for tool I/O: {MAX_METADATA_STRING_LENGTH}")

    qdrant_client = None
    logger.info("DEBUG: Attempting Qdrant client setup...") # New log
    try:
        if _project_modules_available:
            qdrant_client = get_qdrant_client() # Reads QDRANT_URL and QDRANT_API_KEY from env
            await ensure_qdrant_collection_async(qdrant_client, QDRANT_COLLECTION_NAME, VECTOR_DIM)
            logger.info(f"Ensured Qdrant collection '{QDRANT_COLLECTION_NAME}' exists with vector size {VECTOR_DIM}.")
        else:
            logger.warning("Project Qdrant modules not available. Using mock client. No data will be stored.")
            qdrant_client = get_qdrant_client() # Gets mock client
            # No ensure_collection call needed for mock, or implement a mock ensure_collection
        logger.info("DEBUG: Qdrant client setup successful.") # New log
    except Exception as e_setup:
        logger.error(f"Error during Qdrant setup: {e_setup}. Exiting.")
        return

    conversations_data = None # Initialize to check if loaded
    logger.info(f"DEBUG: Attempting to load conversations file: {CONVERSATIONS_FILE_PATH}") # New log
    try:
        with open(CONVERSATIONS_FILE_PATH, 'r', encoding='utf-8') as f:
            conversations_data = json.load(f)
        logger.info(f"DEBUG: Successfully loaded and parsed {CONVERSATIONS_FILE_PATH}.") # New log
    except FileNotFoundError:
        logger.error(f"Conversations file not found at {CONVERSATIONS_FILE_PATH}. Exiting.")
        return
    except json.JSONDecodeError as e_json:
        logger.error(f"Error decoding JSON from {CONVERSATIONS_FILE_PATH}: {e_json}. Exiting.")
        return
    except Exception as e_file:
        logger.error(f"Error reading conversations file: {e_file}. Exiting.")
        return

    if not isinstance(conversations_data, list):
        logger.error("Conversations data is not a list. Expected a list of conversation objects. Exiting.")
        return
    logger.info("DEBUG: Conversations data loaded and validated as list.") # New log

    all_processed_texts_for_embedding: List[str] = []
    all_payload_templates: List[Dict[str, Any]] = []
    total_conversations = len(conversations_data)
    global_successful_points = 0
    global_failed_points = 0

    logger.info("DEBUG: Starting main conversation processing loop...") # New log
    for idx, conversation_obj in enumerate(conversations_data):
        logger.info(f"Processing conversation {idx + 1}/{total_conversations} (UUID: {conversation_obj.get('uuid', 'N/A')}, Name: {conversation_obj.get('name', 'N/A')})")
        
        # segments_data is List[Tuple[str, Dict[str, Any]]]
        # where str is the chunked_text and Dict is its payload_template
        segments_data = process_conversation_for_embedding(conversation_obj)
        
        for chunked_text, payload_template in segments_data:
            all_processed_texts_for_embedding.append(chunked_text)
            all_payload_templates.append(payload_template)

        # Process in batches based on PROCESSING_BATCH_SIZE
        if len(all_processed_texts_for_embedding) >= PROCESSING_BATCH_SIZE or (idx == total_conversations - 1 and all_processed_texts_for_embedding):
            logger.info(f"Preparing to process a batch of {len(all_processed_texts_for_embedding)} text segments for embedding and Qdrant storage.")
            
            current_texts_batch = all_processed_texts_for_embedding
            current_payloads_batch = all_payload_templates
            all_processed_texts_for_embedding = []
            all_payload_templates = []

            try:
                logger.info(f"Requesting embeddings for {len(current_texts_batch)} text segments...")
                # create_embeddings_batch internally uses EMBEDDING_BATCH_SIZE from env (e.g. 512 or default 32)
                # to make multiple HTTP calls if current_texts_batch is larger than that.
                # Corrected call: create_embeddings_batch from embedding_utils.py only takes `texts`
                embeddings = await create_embeddings_batch(current_texts_batch)
                logger.info(f"Received {len(embeddings)} embedding results (includes None for failures).")

                points_to_upsert_master_list: List[PointStruct] = []
                batch_successful_points = 0
                batch_failed_points = 0

                for i, embedding_vector in enumerate(embeddings):
                    if embedding_vector:
                        payload = current_payloads_batch[i].copy()
                        payload["text_content_embedded"] = current_texts_batch[i] # Store the actual chunked text
                        payload["char_count"] = len(current_texts_batch[i])
                        payload["word_count"] = len(current_texts_batch[i].split())
                        # Ensure these are set if not already by specific types
                        payload.setdefault("tool_name", None)
                        payload.setdefault("tool_use_id", None)
                        payload.setdefault("tool_is_error", None)


                        points_to_upsert_master_list.append(
                            PointStruct(
                                id=str(uuid.uuid4()),
                                vector=embedding_vector,
                                payload=payload
                            )
                        )
                        batch_successful_points += 1
                    else:
                        logger.warning(f"Embedding failed for text segment: '{current_texts_batch[i][:100]}...'. Skipping this segment.")
                        batch_failed_points += 1
                
                global_successful_points += batch_successful_points
                global_failed_points += batch_failed_points

                if points_to_upsert_master_list:
                    if _project_modules_available and qdrant_client:
                        try:
                            # Upsert to Qdrant in smaller batches defined by QDRANT_UPSERT_BATCH_SIZE
                            for i in range(0, len(points_to_upsert_master_list), QDRANT_UPSERT_BATCH_SIZE):
                                qdrant_batch_to_upsert = points_to_upsert_master_list[i:i + QDRANT_UPSERT_BATCH_SIZE]
                                try:
                                    await asyncio.to_thread(qdrant_client.upsert, collection_name=QDRANT_COLLECTION_NAME, points=qdrant_batch_to_upsert)
                                    logger.info(f"Successfully upserted {len(qdrant_batch_to_upsert)} points to Qdrant (sub-batch {i // QDRANT_UPSERT_BATCH_SIZE + 1}).")
                                except Exception as e_upsert_sub_batch:
                                    logger.error(f"Error upserting {len(qdrant_batch_to_upsert)} points to Qdrant: {e_upsert_sub_batch}")
                                    global_failed_points += len(qdrant_batch_to_upsert) 
                                    global_successful_points -= len(qdrant_batch_to_upsert)
                        except Exception as e_upsert:
                            logger.error(f"Error upserting {len(points_to_upsert_master_list)} points to Qdrant: {e_upsert}")
                            global_failed_points += len(points_to_upsert_master_list)
                            global_successful_points -= len(points_to_upsert_master_list)
                    elif qdrant_client: # Mock client
                         # Simulate batching for mock client as well
                        for i in range(0, len(points_to_upsert_master_list), QDRANT_UPSERT_BATCH_SIZE):
                            mock_qdrant_batch = points_to_upsert_master_list[i:i + QDRANT_UPSERT_BATCH_SIZE]
                            await qdrant_client.upsert(collection_name=QDRANT_COLLECTION_NAME, points=mock_qdrant_batch)
                            logger.info(f"Mock upserted {len(mock_qdrant_batch)} points (sub-batch {i // QDRANT_UPSERT_BATCH_SIZE + 1}).")
                    else:
                        logger.error("Qdrant client not available. Cannot upsert points.")

                logger.info(f"Batch processing complete. Successful embeddings for this batch: {batch_successful_points}, Failed embeddings: {batch_failed_points}")

            except Exception as e_batch_proc:
                logger.error(f"Error during batch processing (embedding or Qdrant upsert): {e_batch_proc}")
                global_failed_points += len(current_texts_batch) # Assume all in this batch failed at some point

    # Final summary log
    logger.info("Claude conversation history ingestion finished.")
    logger.info(f"Total points attempted: {global_successful_points + global_failed_points}")
    logger.info(f"Successfully embedded and attempted Qdrant upsert: {global_successful_points}")
    logger.info(f"Failed to embed or upsert: {global_failed_points}")


if __name__ == "__main__":
    # Ensure the script can be run with `python -m scripts.ingest_claude_history`
    # For that, the parent directory of `scripts` (i.e., the project root) must be in PYTHONPATH
    # This is often handled by IDEs or by setting PYTHONPATH manually.
    # If running directly `python scripts/ingest_claude_history.py`, relative imports
    # for `src.*` might fail unless the project root is added to sys.path.
    # The try-except for imports helps manage this.
    
    # Add project root to sys.path if not already there, to help with `src` imports
    # when running as `python scripts/ingest_claude_history.py`
    import sys
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # Initialize logger here if it's not already (it should be by now due to global scope)
    # This is to ensure logger.debug works in this block if needed.
    _main_logger = get_logger(__name__ + ".main_runner") # Use a distinct name for this specific logger if needed

    if project_root not in sys.path:
        sys.path.insert(0, project_root)
        _main_logger.info(f"DEBUG: Added project root {project_root} to sys.path for module resolution.")
    else:
        _main_logger.info(f"DEBUG: Project root {project_root} already in sys.path.")
    
    _main_logger.info("DEBUG: Attempting to run main() via asyncio.run()...")
    try:
        asyncio.run(main())
        _main_logger.info("DEBUG: asyncio.run(main()) completed.")
    except Exception as e_async_run:
        _main_logger.error(f"CRITICAL ERROR during asyncio.run(main()): {e_async_run}", exc_info=True)
    finally:
        _main_logger.info("DEBUG: Script execution finished or terminated.") 