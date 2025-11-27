from typing import Dict, Any, List
import json
import asyncio
from datetime import datetime
import poml
import pypdf

try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import (
        AzureTextEmbedding, 
        OpenAIEmbeddingPromptExecutionSettings,
        AzureChatCompletion,
        AzureChatPromptExecutionSettings
    )
    from semantic_kernel.functions import KernelArguments
    from semantic_kernel.contents import ChatHistory
    from dataclasses import dataclass
    from typing import Annotated
    from settings_config import AppSettings
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False


from china_compliance_utils import (
        get_default_rag_settings,
        get_default_model_parameters,
        create_chroma_store_collection,
        load_and_embed_pdf_data,
        should_load_and_embed_pdf_data,
        prepare_product_context
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text content from a PDF file."""
    try:
        with open(pdf_path, "rb") as file:  # Note: "rb" for binary mode
            pdf_reader = pypdf.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""

async def main():
    settings = AppSettings()
    headers =  {
        "client_id": settings.nesgen_client_id,
        "client_secret": settings.nesgen_client_secret,
    } if (settings.nesgen_client_id  and settings.nesgen_client_secret) else None   
    chat_completion_service = AzureChatCompletion(
                    api_key=settings.azure_openai_api_key,
                    endpoint=settings.azure_openai_endpoint,
                    deployment_name=settings.azure_openai_deployment_name,
                    api_version=settings.azure_openai_api_version,
                    default_headers=headers
            )
    chat_settings = AzureChatPromptExecutionSettings(
        temperature=1,
        top_p=1
    )
    embedding_service = AzureTextEmbedding(
                api_key=settings.azure_openai_api_key,
                endpoint=settings.azure_openai_endpoint,
                deployment_name=settings.azure_openai_embeddings_deployment_name,
                api_version=settings.azure_openai_embeddings_api_version,
                default_headers=headers
            )
    kernel = sk.Kernel()
    kernel.add_service(chat_completion_service)
    kernel.add_service(embedding_service)
    rag_settings = get_default_rag_settings()
            
            # Initialize ChromaDB vector store
    chroma_store, pdf_collection = create_chroma_store_collection(
        embedding_service, rag_settings
    )
            
            # Load and embed PDF data if needed
    if await should_load_and_embed_pdf_data(chroma_store, pdf_collection):
        await load_and_embed_pdf_data(chroma_store, rag_settings)

    try:
        search_function = pdf_collection.create_search_function(
            function_name="recall_pdf_info",
            description="Recalls relevant information from PDF documents.",
            string_mapper=lambda result: result.record.text,
        )
        kernel.add_function(
            "PdfMemory",
            search_function,
        )
        print("Successfully registered RAG function with kernel")
    except Exception as e:
        print(f"Error registering RAG function: {e}")
    
    try:
        recall_function = kernel.plugins["PdfMemory"]["recall_pdf_info"]
        if not recall_function:
            return "Regulatory search function not available."
    except (KeyError, AttributeError):
            return "Regulatory search function not properly registered."
            
    raw_search_result = await kernel.invoke(recall_function, arguments=KernelArguments(
        query=f"Get all Promotion rules",
        limit=10,
        min_relevance=0.7
    ))

    # retrieved_context = str(raw_search_result)
    # retrieved_context = extract_text_from_pdf("documents/regulations_chinese.pdf")
    retrieved_context = None
    with open("prompts/generated_translated_context.txt", "r") as f:
        retrieved_context = f.read()

    chat_history = ChatHistory()
    with open("prompts/system_conversation_generation_prompt.txt", "r") as f:
        sys_prompt = f.read()
        sys_prompt = sys_prompt.replace("{{regulation_context}}", retrieved_context)
        # print(sys_prompt)
        chat_history.add_system_message(sys_prompt)

    CATEGORY_PROMPTS = {
        "CHECKED": "Create an example where the marketing manager provides complete campaign details that align with all regulatory requirements.",
        "INVALID": "Create an example where the marketing manager unknowingly overlooks a compliance requirement in their campaign planning.",   
        "CHECK_FAILED": "Create an example where the marketing manager hasn't yet provided all the necessary campaign information."
    }
    right_key = "CHECKED"
    idx = 0
    chat_history.add_user_message(CATEGORY_PROMPTS[right_key])
    response_object = await chat_completion_service.get_chat_message_contents(
        chat_history=chat_history,
        settings=chat_settings
    )
    llm_response = response_object[0].content
    try:
        llm_json = json.loads(llm_response)
        
        json_text = json.dumps({"conversation": llm_json.get("conversation")}, indent=4, ensure_ascii=False)
        with open(f"dataset/inputs/inputs{idx:03d}.json", "w", encoding="utf-8") as f:
            f.write(json_text)
        
        res_json = {"context": llm_json.get("context"), "compliance": right_key}
        json_res = json.dumps(res_json, indent=4, ensure_ascii=False)
        with open(f"dataset/results/results{idx:03d}.json", "w", encoding="utf-8") as f:
            f.write(json_res)
        
    except:
        print("NUUUU A MERS CONVERSIA IN JSON")



if __name__ == "__main__":
    asyncio.run(main())