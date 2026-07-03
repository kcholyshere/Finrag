from dotenv import load_dotenv
from langfuse import get_client

load_dotenv()

get_langfuse_client = get_client

