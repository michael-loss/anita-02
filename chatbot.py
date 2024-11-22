import boto3
import json
import os
import streamlit as st
import toml
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth

class ChatHandler:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def add_message(self, role, content):
        if role == "human":
            self.memory.chat_memory.add_message(HumanMessage(content=content))
        elif role == "ai":
            self.memory.chat_memory.add_message(AIMessage(content=content))

    def get_chat_history(self):
        return self.memory.chat_memory.messages

    def get_conversation_string(self):
        messages = self.get_chat_history()
        return "\n".join([f"{msg.type}: {msg.content}" for msg in messages])

def load_streamlit_secrets():
    """
    Load environment variables from Streamlit secrets
    """
    try:
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') is not None
        
        if is_streamlit_cloud:
            for key, value in st.secrets.items():
                if not key.startswith('_'):
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            full_key = f"{key}_{sub_key}".upper()
                            os.environ[full_key] = str(sub_value)
                    else:
                        os.environ[key.upper()] = str(value)
            return True
        
        else:
            secrets_path = Path('.streamlit/secrets.toml')
            
            if not secrets_path.exists():
                st.warning(f"Warning: {secrets_path} not found")
                return False
                
            secrets = toml.load(secrets_path)
            
            for key, value in secrets.items():
                if isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        full_key = f"{key}_{sub_key}".upper()
                        os.environ[full_key] = str(sub_value)
                else:
                    os.environ[key.upper()] = str(value)
            
            return True
            
    except Exception as e:
        st.error(f"Error loading secrets: {str(e)}")
        return False

def initialize_aws_clients():
    """
    Initialize AWS clients with error handling
    """
    try:
        load_streamlit_secrets()
        
        session = boto3.Session(
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        
        bedrock = session.client('bedrock-runtime', 'us-east-1')
        opensearch_client = session.client("opensearchserverless")
        
        return session, bedrock, opensearch_client
    
    except Exception as e:
        st.error(f"AWS Client Initialization Error: {e}")
        return None, None, None

def answer_query(user_input, chat_session, host, region='us-east-1', index='embeddings'):
    try:
        # Get chat history for context
        chat_history = chat_session.get_chat_history()
        
        # Format user input with chat history context
        context = "\n".join([str(message) for message in chat_history])
        enhanced_query = f"Context from previous conversation:\n{context}\n\nCurrent question: {user_input}"
        
        # Initialize Bedrock client
        bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region
        )
        
        # Format prompt for Claude 3.5 Sonnet
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": enhanced_query
                }
            ],
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 250,
            "stop_sequences": ["Human:", "Assistant:"]
        }

        # Generate response using Claude 3.5 Sonnet
        response = bedrock.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20240620-v1:0",
            contentType="application/json",
            accept="application/json",
            body=json.dumps(request_body)
        )
        
        response_body = json.loads(response.get('body').read())
        ai_response = response_body.get('content')[0].get('text', '').strip()

        # Save the interaction to chat memory
        chat_session.save_message(user_input, ai_response)
        
        return ai_response

    except Exception as e:
        error_message = f"Error in answer_query: {str(e)}"
        print(error_message)
        return error_message


    except Exception as e:
        st.error(f"Query processing error: {e}")
        return "An error occurred while processing your query."