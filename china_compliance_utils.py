import os
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import sys
from pypdf import PdfReader

# Import core Semantic Kernel components that are always needed
from semantic_kernel.functions import KernelArguments

try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureTextEmbedding
    from dataclasses import dataclass
    from semantic_kernel.connectors.chroma import ChromaStore, ChromaCollection
    from semantic_kernel.data.vector import VectorStoreField, vectorstoremodel
    from typing import Annotated
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False

if SK_AVAILABLE:
    @vectorstoremodel(collection_name="pdf_chunks")
    @dataclass
    class PdfChunk:
        id: Annotated[str, VectorStoreField("key")]
        text: Annotated[str, VectorStoreField("data")]
        embedding: Annotated[
            list[float] | str | None,
            VectorStoreField("vector", dimensions=1536),
        ] = None
        page_number: Annotated[int | None, Any] = None
        source_file: Annotated[str | None, Any] = None
        chunk_index: Annotated[int | None, Any] = None

        def __post_init__(self):
            if self.embedding is None:
                self.embedding = self.text

def get_default_rag_settings() -> Dict[str, Any]:
    """Get default RAG settings for compliance checking"""
    return {
        "collection_name": "compliance_pdf_collection",
        "max_retrieved_chunks": 5,
        "min_relevance_score": 0.7,
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

def get_default_model_parameters() -> Dict[str, Any]:
    """Get default model parameters for AI generation"""
    return {
        "temperature": 0.5,
        "top_p": 0.5
    }

def create_search_arguments_structure(ingredients_str: str, allergens_str: str, 
                                    user_input: str, rag_settings: Dict[str, Any] = None) -> KernelArguments:
    """Create search arguments for RAG queries"""
    if rag_settings is None:
        rag_settings = get_default_rag_settings()
        
    rag_query_text = (
        f"Ingredients: {ingredients_str}. "
        f"Declared Allergens: {allergens_str}. "
        f"User query: {user_input}"
    )
    
    # Configure vector search parameters
    search_arguments = KernelArguments(
        query=rag_query_text,
        limit=rag_settings.get("max_retrieved_chunks", 5),
        min_relevance=rag_settings.get("min_relevance_score", 0.7),
    )
    
    return search_arguments

def create_chroma_store_collection( embedding_service_instance : AzureTextEmbedding, 
                                   rag_settings) -> tuple[ChromaStore, ChromaCollection[str, PdfChunk]]: 

  # Initialize Chroma vector store with the embedding generator
  chroma_store_connector = ChromaStore(
    persist_directory=os.getenv('CHROMA_PERSIST_DIR'),
    embedding_generator=embedding_service_instance,
  )
  print("ChromaStore (ChromaDB) initialized and connected to embedding service.")

  # Obtain a typed Chroma collection for PdfChunk records
  pdf_collection: ChromaCollection[str, PdfChunk] = (
    chroma_store_connector.get_collection(
      record_type=PdfChunk, collection_name=rag_settings.get("collection_name", "compliance_pdf_collection")
    )
  )
 
  return chroma_store_connector, pdf_collection

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
  """Split text into overlapping chunks by approximate character count.

  Strategy:
  - Split by words to avoid hard-cutting inside words.
  - Build chunks that do not exceed chunk_size (approx).
  - Create overlap between chunks of up to chunk_overlap characters
    by carrying trailing words.

  Notes:
  - This is a character-length heuristic using word lengths; not exact.
  - Overlap improves RAG recall by preserving context across boundaries.
  """
  chunks = []
  if not text:
    return chunks

  words = text.split()
  current_chunk_words = []
  current_chunk_length = 0

  for word in words:
    # If adding this word would exceed chunk_size (plus space),
    # finalize the current chunk and start a new one with overlap.
    if (current_chunk_length + len(word) + 1 > chunk_size and current_chunk_words):
      chunks.append(" ".join(current_chunk_words).strip())

      # Compute overlap by walking backward and accumulating words until
      # we hit the overlap character budget.
      overlap_words_for_next_chunk = []
      overlap_char_count = 0
      for w in reversed(current_chunk_words):
        if overlap_char_count + len(w) + 1 <= chunk_overlap:
          overlap_words_for_next_chunk.insert(0, w)
          overlap_char_count += len(w) + 1
        else:
          break

      current_chunk_words = overlap_words_for_next_chunk
      current_chunk_length = (
        sum(len(w) for w in current_chunk_words)
        + (len(current_chunk_words) - 1 if len(current_chunk_words) > 0 else 0)
      )

    # Add the current word and update current length (+1 for space)
    current_chunk_words.append(word)
    current_chunk_length += len(word) + 1

  # Flush remainder
  if current_chunk_words:
    chunks.append(" ".join(current_chunk_words).strip())
  return chunks

async def should_load_and_embed_pdf_data(store : ChromaStore,
    pdf_collection : ChromaCollection[str, PdfChunk]) -> bool:
    try:
      results = await pdf_collection._inner_get(
            options=None,
            include_vectors=True,
        )
      return False if results else True
    except Exception as e:
      return True
    

def extract_raw_text_from_pdf(pdf_filepath : str) -> str:
  full_text = ""
  try:
    reader = PdfReader(pdf_filepath)
    num_pages = len(reader.pages)
    print(f"Reading PDF: '{pdf_filepath}' with {num_pages} pages...")
    for i, page in enumerate(reader.pages):
      page_text = page.extract_text()
      if page_text:
        full_text += page_text + "\n\n"
      # Progress logging every 10 pages for long PDFs
      if (i + 1) % 10 == 0:
        print(f"  Processed {i+1}/{num_pages} pages...")
    print("PDF text extraction complete.")
    return full_text
  except Exception as e:
    sys.exit(f"Error extracting text from PDF '{pdf_filepath}': {e}")


async def load_and_embed_pdf_data(
  chroma_store_instance: ChromaStore,
  rag_settings : dict,
):
  """Pipeline: read PDF -> extract text -> chunk -> upsert into Chroma with embeddings.

  Key points:
  - Uses pypdf to extract text; quality depends on PDF structure (may be empty for
    scanned PDFs without OCR).
  - Chunks are created to support RAG (retrieval augmented generation).
  - Upsert in batches to the specified Chroma collection; embeddings generated via
    SK embedding service configured on ChromaStore.
  """
  # print("\n--- Phase 1: Loading and Embedding PDF Data into Vector DB ---")

  collection_name = rag_settings.get("collection_name", "compliance_pdf_collection")
  chunk_size = rag_settings.get("chunk_size", 1000)
  chunk_overlap = rag_settings.get("chunk_overlap", 200)
  files_directory = os.getenv('INFO_LOCATION_DIR')
  # Upsert into ChromaDB collection with auto-embedding
  try:
    collection: ChromaCollection[str, PdfChunk] = (
      chroma_store_instance.get_collection(
        record_type=PdfChunk, collection_name=collection_name
      )
    )

    # Ensure physical collection exists on disk/backing store
    await collection.ensure_collection_exists()
  
    for source_file_name in os.listdir(files_directory):
      pdf_filepath = os.path.join(files_directory, source_file_name)
      if not os.path.exists(pdf_filepath):
        sys.exit(f"Error: PDF file not found at '{pdf_filepath}'.") 
      full_text = extract_raw_text_from_pdf(pdf_filepath)
    
      chunks = chunk_text(full_text, chunk_size, chunk_overlap)
      if not chunks:
        sys.exit("No text chunks generated from PDF.")
      print(f"Split PDF into {len(chunks)} chunks for embedding.")
    
      # 3) Wrap chunks into PdfChunk records with metadata
      pdf_chunk_items = [
        PdfChunk(
          id=f"{source_file_name.replace('.', '_')}_chunk_{i}",
          text=chunk_text_content,
          # page_number is not tracked per chunk here; set to 1 as placeholder
          page_number=1,
          source_file=source_file_name,
          chunk_index=i
        ) for i, chunk_text_content in enumerate(chunks)
      ]
      print(
        f"Upserting {len(pdf_chunk_items)} records to collection '{collection_name}'..."
      )

      # Batch size to avoid large payloads/timeouts; tune if needed
      BATCH_SIZE = 512
      total_upserted = 0

      for i in range(0, len(pdf_chunk_items), BATCH_SIZE):
        batch = pdf_chunk_items[i : i + BATCH_SIZE]
        print(
          f"  Upserting batch {i//BATCH_SIZE + 1}/"
          f"{(len(pdf_chunk_items) + BATCH_SIZE - 1) // BATCH_SIZE} "
          f"({len(batch)} items)..."
        )
        await collection.upsert(batch)
        total_upserted += len(batch)
        print(
         f"  Successfully upserted {total_upserted}/{len(pdf_chunk_items)} records so far."
        )

      print(f"\nSuccessfully upserted {total_upserted} records in total.")

  except Exception as e:
    # Surface the error and re-raise to fail fast in calling context
    sys.exit(f"Error during data ingestion: {e}")


def prepare_product_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare product context for compliance checking from workflow context"""
    return {
        'product_name': context.get('project_name', 'Unknown Product'),
        'ingredients': context.get('ingredients', []),
        'allergens': context.get('allergens', []),
        'markets': context.get('market_selections', []),
        'brands': context.get('brand_selections', []),
        'skus': context.get('sku_selections', []),
        'changes': context.get('change_description', ''),
        'success_criteria': context.get('success_criteria', [])
    }

def extract_compliance_status(compliance_answer: str) -> str:
    """Extract overall compliance status from the AI response"""
    if 'Overall Status: COMPLIANT' in compliance_answer:
        return 'COMPLIANT'
    elif 'Overall Status: NON-COMPLIANT' in compliance_answer:
        return 'NON-COMPLIANT'
    elif 'Overall Status: UNKNOWN_COMPLIANCE_STATUS' in compliance_answer:
        return 'UNKNOWN_COMPLIANCE_STATUS'
    else:
        # Try to infer from the content
        if 'COMPLIANT' in compliance_answer.upper():
            return 'COMPLIANT'
        elif 'NON-COMPLIANT' in compliance_answer.upper():
            return 'NON-COMPLIANT'
        else:
            return 'UNKNOWN_COMPLIANCE_STATUS'
        
def extract_summary(compliance_answer: str) -> str:
    """Extract a summary from the compliance answer"""
    # Take the first few sentences as summary
    sentences = compliance_answer.split('.')
    if len(sentences) > 0:
        summary = sentences[0].strip()
        if len(summary) > 100:
            summary = summary[:97] + "..."
        return summary
    return "Compliance assessment completed"