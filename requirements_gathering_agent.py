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

class RequirementsGatheringAgent:
    step_name = "reguirements_gathering_agent"
    description = ""
    field_name = ""
    validation_rules = [""]

    def __init__(self):
        self.kernel = None
        self.chat_completion_service = None
        self.chat_settings = None
        self.pdf_collection = None
        self.initialized = False

    async def initialize_services(self):
        settings = AppSettings()
        """Initialize the semantic kernel and compliance services"""
        if self.initialized:
            return True

        try:
            # Check if we have the required environment variables
            if not settings.azure_openai_api_key or not settings.azure_openai_endpoint:
                print("Missing Azure OpenAI credentials in environment variables")
                return False

            headers =  {
                    "client_id": settings.nesgen_client_id,
                    "client_secret": settings.nesgen_client_secret,
            } if (settings.nesgen_client_id  and settings.nesgen_client_secret) else None   

            # Initialize Azure OpenAI services
            embedding_service = AzureTextEmbedding(
                api_key=settings.azure_openai_api_key,
                endpoint=settings.azure_openai_endpoint,
                deployment_name=settings.azure_openai_embeddings_deployment_name,
                api_version=settings.azure_openai_embeddings_api_version,
                default_headers=headers
            )

            self.chat_completion_service = AzureChatCompletion(
                    api_key=settings.azure_openai_api_key,
                    endpoint=settings.azure_openai_endpoint,
                    deployment_name=settings.azure_openai_deployment_name,
                    api_version=settings.azure_openai_api_version,
                    default_headers=headers
            )
            self.chat_settings = AzureChatPromptExecutionSettings(
                temperature=0.1,
                top_p=0.8
            )
            
            # Initialize Semantic Kernel
            self.kernel = sk.Kernel()
            self.kernel.add_service(self.chat_completion_service)
            self.kernel.add_service(embedding_service)

            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing compliance services: {e}")
            return False
        
    def get_system_prompt_info_gathering(self, product_context : Dict[str, Any], context : Dict[str, Any]):
        poml_params = poml.poml(
        "system_prompt_info_gathering.poml",
            context={
                "product_context": product_context,
                "promotion_context": context,
            },
            format="openai_chat"
        )
        messages = poml_params.get("messages", [])
        if messages:
            return messages[0].get("content", "")
        return ""

    async def gather_context_information(self, product_context : Dict[str, Any], context : Dict[str, Any],
                                          inputs : list[str]):
        system_prompt = self.get_system_prompt_info_gathering(product_context, context)
        try:
            if not await self.initialize_services():
                return self.step_name, {
                    **context,
                    'error': 'Compliance services not available. Please check configuration.'
                }
            chat_history = ChatHistory()
            chat_history.add_system_message(system_prompt)
            for input in inputs:
                chat_history.add_user_message(input)
                response_object = await self.chat_completion_service.get_chat_message_contents(
                    chat_history=chat_history,
                    settings=self.chat_settings
                )
                chat_history.add_assistant_message(response_object[0].content)
                llm_response =  response_object[0].content
                try: 
                    json_response = json.loads(llm_response)
                    print(json_response.keys())
                    print(json_response.get("status"))
                    print(json_response.get("question"))
                    print(json_response.get("promotion_context"))
                    print("\n\n\n")
                except:
                    "THIS IS NOT A JSON WHYYYYYY?????"
            return chat_history.messages[-1].content

        except Exception as e:
            error_msg = f"Error processing compliance request: {str(e)}"
            self.add_to_conversation_history(f"System Error: {error_msg}")
            return self.step_name, {
                **context,
                'error': error_msg
            }

from utils_requirements_gathering_agent import (
    read_dataset,
    compare_contexts
)


async def main():
    requirementsGatheringAgent = RequirementsGatheringAgent()
    sample_context = {
        'project_name': 'Nescafé Gold Blend European Launch',
        'market_selections': ["China Market"],
        'brand_selections': ['Nestle Kitkat'],
        'sku_selections': ['NES-GB-200G'],
        'change_description': """Renovation requrires GTIN changes, however since it is not seen as a new product this will be classified as a Normal project with no GTIN change""",
        'success_criteria': ['Ensure regulatory compliance'],
    }
    context =  {
        "promotion_name": None,
        "promotion_type": None,       
        "applicable_products": None,  
        "participation_conditions": None,
        "prizes": None,                   
        "prizez_number_of_winners": None,        
        "draw_method": None,           
        "draw_time": None,
        "duration": {
            "start_date": None,
            "end_date": None
        },
        "promotion_rules_link": None,       
        "promotion_qr_code": None,     
        "personal_information_consent": None, 
        "reference_price": None,       
        "discount_amount": None,
        "redemption_method": None,     
        "other_restrictions": None     
    }

    train_data, results = read_dataset()
    for (train_conv, result) in zip(train_data, results):
        llm_context = await requirementsGatheringAgent.gather_context_information(sample_context, context, train_conv)
        groundtruth_context = result.get("context")
        print(compare_contexts(groundtruth_context, json.loads(llm_context)))

if __name__ == "__main__":
    asyncio.run(main())
