import os
import PyPDF2
import re
from typing import Dict, Any, List
import json
import asyncio
from datetime import datetime
import poml

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

def extract_text_pdf(filepath):
    text = ""
    with open(filepath, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""
            text += "\n"
    text = re.sub(r'\n\s*\n+', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = '\n'.join(line.strip() for line in text.splitlines())
    text = text.strip()
    return text

# Example
async def main():
    text = extract_text_pdf("documents/regulations_chinese.pdf")
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
   
    chat_history = ChatHistory()
    chat_history.add_system_message("Translate the following chinese document into English.")
    chat_history.add_user_message(text)
    response_object = await chat_completion_service.get_chat_message_contents(
        chat_history=chat_history,
        settings=AzureChatPromptExecutionSettings(
            temperature=0.1,
            top_p=0.8
        )
    )
    with open("prompts/out_translated.txt", "w") as f:
        f.write(response_object[0].content)
    chat_history.add_system_message("I want to use the translated content to pass it as part of system prompt as RAG. " \
    "You have to filter the translated content and keep only information related to promotion regulation, remove information related to penalties. Do NOT invent rules, keep only information from document")
    chat_history.add_user_message(response_object[0].content)
    response_object = await chat_completion_service.get_chat_message_contents(
        chat_history=chat_history,
        settings=AzureChatPromptExecutionSettings(
            temperature=0.1,
            top_p=0.8
        )
    )
    with open("prompts/generated_translated_context.txt", "w") as f:
        f.write(response_object[0].content)


if __name__ == "__main__":
    asyncio.run(main())
    

