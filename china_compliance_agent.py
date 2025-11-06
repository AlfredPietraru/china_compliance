from typing import Dict, Any, List
import json
import asyncio
from datetime import datetime

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

# Import compliance utilities
try:
    from china_compliance_utils import (
        get_default_rag_settings,
        get_default_model_parameters,
        create_search_arguments_structure,
        create_chroma_store_collection,
        load_and_embed_pdf_data,
        should_load_and_embed_pdf_data,
        prepare_product_context,
        extract_compliance_status,
        extract_summary
    )
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False

class ChinaComplianceAgent:
    step_name = "china_regulatory_compliance"
    description = ""
    field_name = ""
    validation_rules = [""]

    def __init__(self):
        self.conversation_history = []
        self.kernel = None
        self.chat_completion_service = None
        self.chat_settings = None
        self.pdf_collection = None
        self.initialized = False

    def add_to_conversation_history(self, message: str):
        """Add a message to the conversation history."""
        self.conversation_history.append(message)
        # Keep only last 10 messages to avoid memory issues
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]

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
            
            # Initialize settings with default parameters
            model_params = get_default_model_parameters()
            self.chat_settings = AzureChatPromptExecutionSettings(
                temperature=model_params.get("temperature", 0.5),
                top_p=model_params.get("top_p", 0.5)
            )
            
            # Get default RAG settings
            rag_settings = get_default_rag_settings()
            
            # Initialize ChromaDB vector store
            chroma_store, self.pdf_collection = create_chroma_store_collection(
                embedding_service, rag_settings
            )
            
            # Load and embed PDF data if needed
            if await should_load_and_embed_pdf_data(chroma_store, self.pdf_collection):
                await load_and_embed_pdf_data(chroma_store, rag_settings)
            
            # Initialize Semantic Kernel
            self.kernel = sk.Kernel()
            self.kernel.add_service(self.chat_completion_service)
            self.kernel.add_service(embedding_service)
            
            # Add RAG function for PDF retrieval
            try:
                search_function = self.pdf_collection.create_search_function(
                    function_name="recall_pdf_info",
                    description="Recalls relevant information from PDF documents.",
                    string_mapper=lambda result: result.record.text,
                )
                self.kernel.add_function(
                    "PdfMemory",
                    search_function,
                )
                print("Successfully registered RAG function with kernel")
            except Exception as e:
                print(f"Error registering RAG function: {e}")
                # Continue without RAG function for now
            
            self.initialized = True
            return True
            
        except Exception as e:
            print(f"Error initializing compliance services: {e}")
            return False
        
    async def retrieve_regulatory_context(self, product_context: Dict[str, Any], question: str) -> str:
        """Retrieve relevant regulatory information using RAG"""
        try:
            if not self.kernel or not self.pdf_collection:
                return "No regulatory documents available for compliance checking."
            
            # Check if RAG function is available
            try:
                recall_function = self.kernel.plugins["PdfMemory"]["recall_pdf_info"]
                if not recall_function:
                    return "Regulatory search function not available."
            except (KeyError, AttributeError):
                return "Regulatory search function not properly registered."
            
            # Create search query from product context
            ingredients_str = ", ".join(product_context.get('ingredients', [])) if product_context.get('ingredients') else "No ingredients provided"
            allergens_str = ", ".join(product_context.get('allergens', [])) if product_context.get('allergens') else "No allergens declared"
            
            # Get default RAG settings
            rag_settings = get_default_rag_settings()
            
            # Create search arguments
            search_arguments = create_search_arguments_structure(
                ingredients_str, allergens_str, question, rag_settings
            )
            
            # Search for relevant regulations
            raw_search_result = await self.kernel.invoke(recall_function, arguments=search_arguments)
            retrieved_context = str(raw_search_result)
            
            if not retrieved_context.strip() or "No relevant documents found" in retrieved_context:
                return "No relevant regulations found in the knowledge base."
            
            return retrieved_context
            
        except Exception as e:
            return f"Error retrieving regulatory context: {str(e)}"

    async def generate_compliance_assessment(self, product_context: Dict[str, Any], question: str, regulatory_context: str) -> str:
        """Generate compliance assessment using AI"""
        try:
            if not self.chat_completion_service:
                return "Compliance service not available."
            
            # Create system prompt
            system_prompt = self.create_system_prompt(product_context, regulatory_context, question)
            
            # Create chat history
            chat_history = ChatHistory()
            chat_history.add_system_message(system_prompt)
            chat_history.add_user_message(question)
            
            # Get AI response
            response_object = await self.chat_completion_service.get_chat_message_contents(
                chat_history=chat_history,
                settings=self.chat_settings
            )
            
            return response_object[0].content
            
        except Exception as e:
            return f"Error generating compliance assessment: {str(e)}"

    def create_system_prompt(self, product_context, regulatory_context, question) -> str:
        markets_str = ", ".join(product_context.get('markets', [])) if product_context.get('markets') else "Not specified"
        return f"""
            You are a precise and diligent compliance officer specializing in food packaging regulations.
            Your primary knowledge base is **STRICTLY LIMITED** to the following internal rules.
            You **MUST FIRST ATTEMPT** to answer the user's questions and assess product compliance based **EXCLUSIVELY** on these rules.
            **DO NOT FABRICATE OR INVENT** any rules or rule numbers.
            However, if, after consulting the provided rules, you **cannot definitively determine compliance** for a specific ingredient, component, or product characteristic, you are authorized to use your **general knowledge of food compliance principles** to make a reasoned judgment. Clearly state when you are relying on general knowledge rather than specific quoted rules.
            **Product Details for Assessment:**
            Product Name: {product_context.get('product_name', 'Unknown')}
            Markets: {markets_str}

            **User's Specific Query/Focus for Assessment:**
            {question}

            **When assessing product compliance:**
            - Focus on identifying if each ingredient, component, or characteristic mentioned in the provided product details or the user's query is explicitly *permitted*, *prohibited*, or has *specific conditions/limits* within the provided rules.
            - Pay special attention to the declared allergens and ensure they align with the rules and the ingredient list if applicable.
            - Consider market-specific requirements for the selected markets.
            - If an item is not mentioned in the rules, but you have general knowledge of its compliance, use that, and state it clearly.
            - If compliance cannot be definitively assessed by either the provided rules or your general knowledge, you must state that compliance cannot be determined.

            Follow this exact output format:
            **Product and Components to Assess:**
            {product_context.get('product_name', 'Unknown')} (with focus on: {question})

            **Compliance Assessment:**
            First, for each relevant component or characteristic, identify and **quote the exact relevant rule(s)** from the provided list that apply. If no explicit rule is found, state that. If you are using general knowledge, state that.

            Then, for each aspect assessed, declare if it is 'COMPLIANT', 'NON-COMPLIANT', or 'UNKNOWN_COMPLIANCE_STATUS' based on the assessment. Provide clear reasoning.
            **Overall Compliance Status (if determinable):**
            [Overall Status: COMPLIANT/NON-COMPLIANT/UNKNOWN_COMPLIANCE_STATUS]

            --- RULES ---
            {regulatory_context}
            """


    async def process_input(self, user_input, context):
        """Process user input for regulatory compliance"""
        start_time = datetime.now()
        
        try:
            # Initialize services if not already done
            if not await self.initialize_services():
                return 'regulatory_compliance', {
                    **context,
                    'error': 'Compliance services not available. Please check configuration.'
                }
            
            # Extract user message and action
            action = user_input.get('action')
            user_message = user_input.get('text', '') or user_input.get("question", None)
            if isinstance(user_message, list):
                user_message = ', '.join(user_message)
            
            # Add to conversation history
            self.add_to_conversation_history(f"User: {user_message}")
            
            # Prepare product context for compliance checking
            product_context = prepare_product_context(context)
            
            # If user provided a specific question, use it; otherwise, use default
            if not user_message or user_message.strip() == '':
                user_message = "Assess regulatory compliance for this product across all selected markets"
            
            # Only perform heavy assessment if user clicked Assess, otherwise allow skip or default minimal flow
            perform_assessment = (action == 'assess_compliance') or not action
            regulatory_context = ""
            compliance_answer = ""
            if perform_assessment:
                # Retrieve regulatory context using RAG
                regulatory_context = await self.retrieve_regulatory_context(product_context, user_message)
                # Generate compliance assessment
                compliance_answer = await self.generate_compliance_assessment(product_context, user_message, regulatory_context)
            else:
                compliance_answer = "Compliance assessment deferred by user; proceeding to next step."
            
            # Calculate processing time
            end_time = datetime.now()
            processing_time = int((end_time - start_time).total_seconds() * 1000)
            
            # Prepare compliance assessment data
            compliance_assessment = {
                'answer': compliance_answer,
                'relevant_regulations': [{
                    'text': regulatory_context[:500] + "..." if len(regulatory_context) > 500 else regulatory_context,
                    'relevance_score': 0.85,
                    'source': 'regulations_chinese.pdf'
                }] if regulatory_context and "No relevant regulations found" not in regulatory_context else [],
                'product_context': product_context,
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': processing_time,
                'overall_status': extract_compliance_status(compliance_answer),
                'summary': extract_summary(compliance_answer),
                'risk_level': 'LOW' if extract_compliance_status(compliance_answer) == 'COMPLIANT' else 'HIGH',
            }
            
            # Update context with compliance assessment
            updated_context = {
                **context,
                'compliance_assessment': compliance_assessment,
                'compliance_status': compliance_assessment['overall_status'],
                'conversational_response': compliance_answer
            }
            
            # Add response to conversation history
            self.add_to_conversation_history(f"Assistant: {compliance_answer}")
            return 'artwork_brief_creation', updated_context
                
        except Exception as e:
            error_msg = f"Error processing compliance request: {str(e)}"
            self.add_to_conversation_history(f"System Error: {error_msg}")
            return 'regulatory_compliance', {
                **context,
                'error': error_msg
            }

    def is_satisfied(self, context):
        """Check if compliance requirements are satisfied"""
        compliance_assessment = context.get('compliance_assessment', {})
        overall_status = compliance_assessment.get('overall_status', '')
        return overall_status in ['COMPLIANT', 'NON-COMPLIANT']
    
async def main():
    compliance_agent = ChinaComplianceAgent()
    result = await compliance_agent.initialize_services()
    sample_context = {
        'project_name': 'Nescafé Gold Blend European Launch',
        'market_selections': ["China Market"],
        'brand_selections': ['Nestle Kitkat'],
        'sku_selections': ['NES-GB-200G'],
        'change_description': """Renovation requrires GTIN changes, however since it is not seen as a new product this will be classified as a Normal project with no GTIN change""",
        'success_criteria': ['Ensure regulatory compliance'],
    }
    user_input = {
    "action": "assess_compliance",
    "text": """
    I want to add a promotional element to the side flap with following details and a QR code of the website
    "During the event period, users who purchase event products and use WeChat's "Scan" function to scan the QR code on the product packaging can participate in the event and have a chance to win the following prizes:
    First Prize:
    Apple iPhone 14 Pro Max (Full Netcom 5G version, 512GB), market reference price: ¥11,000.
    Number of winners: 20.
    Second Prize:
    500 yuan JD.com e-gift card, market reference price: ¥500.
    Number of winners: 1,000.
    Third Prize:
    Xiaomi NFC wristband, market reference price: ¥249.
    Number of winners: 2,000.
    Fourth Prize:
    Tencent Video, Mango TV, Bilibili membership discount coupons.
    Number of winners: 2,000,000.
    (Coupons will be displayed in the event page after winning.)
    Fifth Prize:
    Points for small treasure boxes, unlimited quantity.
    Event Period:
    From August 15, 2023, to December 31, 2024, 24:00."""
    }
    _, updated_context = await compliance_agent.process_input(user_input, sample_context)
    print(updated_context)

if __name__ == "__main__":
    asyncio.run(main())
