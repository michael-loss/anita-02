import boto3
import json
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_community.chat_message_histories import ChatMessageHistory
from requests_aws4auth import AWS4Auth
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv()

# Initialize AWS session and clients
session = boto3.Session(
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    region_name=os.getenv('AWS_DEFAULT_REGION')
)

bedrock = session.client('bedrock-runtime', region_name=os.getenv('AWS_DEFAULT_REGION'))

# OpenSearch setup
opensearch = boto3.client("opensearchserverless")
host = os.getenv('opensearch_host')
region = 'us-east-1'
service = 'aoss'
credentials = session.get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=auth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection,
    pool_maxsize=20
)

accept = 'application/json'
contentType = 'application/json'

# ChatHandler class to manage the chat history
class ChatHandler:
    def __init__(self):
        self.memory = ChatMessageHistory()

    def add_message(self, role, content):
        if role == "human":
            self.memory.add_user_message(content)
        elif role == "ai":
            self.memory.add_ai_message(content)

    def get_chat_history(self):
        return self.memory.messages

    def get_conversation_string(self):
        return "\n".join([f"{msg.type}: {msg.content}" for msg in self.memory.messages])

    def save_message(self, user_input, ai_response):
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(ai_response)

# AWS signature for authentication
def get_awsauth(region, service):
    credentials = boto3.Session().get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

# Get embeddings using Bedrock (Titan)
def get_embedding(text):
    request_body = {"inputText": text}
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

# Answer query using Titan LLM model
def answer_query_titan(user_input, chat_handler):
    # Generate embedding for the user input
    userQueryBody = json.dumps({"inputText": user_input})
    userVectors = get_embedding(userQueryBody)

    # Query OpenSearch
    query = {
        "size": 3,
        "query": {
            "knn": {
                "vectors": {
                    "vector": userVectors, "k": 3
                }
            }
        },
        "_source": True,
        "fields": ["text"],
    }

    response = client.search(
        body=query,
        index=os.getenv("vector_index_name")
    )

    similaritysearchResponse = ""
    for i in response["hits"]["hits"]:
        outputtext = i["fields"]["text"]
        similaritysearchResponse += f"Info = {str(outputtext)}\n"

    # Chat history for prompt context
    chat_history = chat_handler.get_conversation_string()

    # Configuring the prompt
    prompt_data = f"""
    Assistant: You are an AI assistant that will help members of the Emergency Nurses Association (ENA) find information about ENA's position statements. Answer the provided question to the best of your ability using the information provided in the Context.
    
    Previous conversation: {chat_history}
    Question: {user_input}

    Here is the text you should use as context: {similaritysearchResponse}

    Assistant:
    """
    
    # Request to the model
    request_body = json.dumps({
        "inputText": prompt_data,
        "textGenerationConfig": {
            "maxTokenCount": 2000,
            "temperature": 0.1,
            "topP": 0.3
        }
    })

    model_id = "amazon.titan-text-premier-v1:0"
    response = bedrock.invoke_model(
        modelId=model_id,
        body=request_body,
        accept=accept,
        contentType=contentType
    )

    response_body = json.loads(response.get('body').read())
    output_text = response_body['results'][0]['outputText']

    chat_handler.add_message("human", user_input)
    chat_handler.add_message("ai", output_text)

    return f"{output_text}\n\nModel used: {model_id}"

# Answer query using Llama model
def answer_query_llama(user_input, chat_handler):
    # Generate embedding for the user input
    userQueryBody = json.dumps({"inputText": user_input})
    userVectors = get_embedding(userQueryBody)

    # Query OpenSearch
    query = {
        "size": 3,
        "query": {
            "knn": {
                "vectors": {
                    "vector": userVectors, "k": 3
                }
            }
        },
        "_source": True,
        "fields": ["text"],
    }

    response = client.search(
        body=query,
        index=os.getenv("vector_index_name")
    )

    similaritysearchResponse = ""
    for i in response["hits"]["hits"]:
        outputtext = i["fields"]["text"]
        similaritysearchResponse += f"Info = {str(outputtext)}\n"

    # Chat history for prompt context
    chat_history = chat_handler.get_conversation_string()

    # Configuring the prompt
    prompt_data = f"""
    Assistant: You are an AI assistant that will help members of the Emergency Nurses Association (ENA) find information about ENA's position statements. Answer the provided question to the best of your ability using the information provided in the Context.

    Previous conversation: {chat_history}
    Question: {user_input}

    Here is the text you should use as context: {similaritysearchResponse}

    Assistant:
    """

    # Request to the model
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>{prompt_data}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    request_body = json.dumps({
        "prompt": formatted_prompt,
        "max_gen_len": 512,
        "temperature": 0.0,
        "top_p": 0.3
    })

    model_id = "meta.llama3-70b-instruct-v1:0"
    response = bedrock.invoke_model(
        modelId=model_id,
        body=request_body,
        accept="application/json",
        contentType="application/json"
    )

    response_body = json.loads(response.get('body').read())
    output_text = response_body.get('generation', '')

    chat_handler.add_message("human", user_input)
    chat_handler.add_message("ai", output_text)

    return f"{output_text}\n\nModel used: {model_id}"
