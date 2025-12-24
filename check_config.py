
import os
from dotenv import load_dotenv
from api.config import is_azure_openai_configured, configs, get_embedder_type, get_embedder_config

load_dotenv()

print(f"AZURE_OPENAI_API_KEY: {os.environ.get('AZURE_OPENAI_API_KEY')}")
print(f"AZURE_OPENAI_ENDPOINT: {os.environ.get('AZURE_OPENAI_ENDPOINT')}")
print(f"AZURE_OPENAI_VERSION: {os.environ.get('AZURE_OPENAI_VERSION')}")

print(f"is_azure_openai_configured: {is_azure_openai_configured()}")

print(f"configs['embedder']: {configs.get('embedder')}")

try:
    print(f"get_embedder_type: {get_embedder_type()}")
except Exception as e:
    print(f"get_embedder_type error: {e}")

try:
    print(f"get_embedder_config: {get_embedder_config()}")
except Exception as e:
    print(f"get_embedder_config error: {e}")
