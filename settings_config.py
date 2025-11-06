import os
from dotenv import load_dotenv

class AppSettings:
    def __init__(self):
        load_dotenv()
        self.azure_openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.azure_openai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_openai_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.azure_openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")
        self.azure_openai_embeddings_deployment_name = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT_NAME")
        self.azure_openai_embeddings_api_version = os.getenv("AZURE_OPENAI_EMBEDDINGS_API_VERSION", "2024-06-01")
        self.nesgen_client_id = os.getenv("NESGEN_CLIENT_ID") or os.getenv("CLIENT_ID")
        self.nesgen_client_secret = os.getenv("NESGEN_CLIENT_SECRET") or os.getenv("CLIENT_SECRET")
