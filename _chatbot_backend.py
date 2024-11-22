from langchain.memory import ConversationBufferMemory
from langchain.schema import messages_from_dict, messages_to_dict
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import json
import os
import toml
from pathlib import Path

class ChatSession:
    def __init__(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
    def save_message(self, user_input, ai_response):
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(ai_response)
        
    def get_chat_history(self):
        return self.memory.load_memory_variables({})["chat_history"]
        
    def clear_memory(self):
        self.memory.clear()

def save_chat_history(chat_session, chat_history_file):
    chat_history = chat_session.get_chat_history()
    chat_history_dict = messages_to_dict(chat_history)
    with open(chat_history_file, 'w') as f:
        json.dump(chat_history_dict, f) 

def load_dotStreat_sl():
    """
    Load environment variables from either:
    1. Streamlit Cloud secrets (if deployed)
    2. Local .streamlit/secrets.toml (if running locally)
    
    Sets values in os.environ for compatibility with existing code.
    
    Returns:
        bool: True if secrets were loaded successfully, False otherwise
    """
    try:
        # Check if running on Streamlit Cloud by looking for STREAMLIT_SHARING_MODE
        is_streamlit_cloud = os.getenv('STREAMLIT_SHARING_MODE') is not None
        
        if is_streamlit_cloud:
            # Running on Streamlit Cloud - use st.secrets
            for key, value in st.secrets.items():
                # Skip internal streamlit keys that start with _
                if not key.startswith('_'):
                    # Handle nested dictionaries in secrets
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            full_key = f"{key}_{sub_key}".upper()
                            os.environ[full_key] = str(sub_value)
                    else:
                        os.environ[key.upper()] = str(value)
            return True
            
        else:
            # Running locally - load from .streamlit/secrets.toml
            secrets_path = Path('.streamlit/secrets.toml')
            
            if not secrets_path.exists():
                print(f"Warning: {secrets_path} not found")
                return False
                
            # Load the TOML file
            secrets = toml.load(secrets_path)
            
            # Add each secret to environment variables
            for key, value in secrets.items():
                if isinstance(value, dict):
                    # Handle nested dictionaries
                    for sub_key, sub_value in value.items():
                        full_key = f"{key}_{sub_key}".upper()
                        os.environ[full_key] = str(sub_value)
                else:
                    os.environ[key.upper()] = str(value)
            
            return True
            
    except Exception as e:
        print(f"Error loading secrets: {str(e)}")
        return False


def get_awsauth(region, service):
    credentials = boto3.Session().get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

def answer_query(user_input, chat_session, host, region='us-east-1', index='embeddings'):
    """
    Takes user question, creates embedding, performs KNN search on OpenSearch index,
    and generates answer using context from similar results.
    """
    # Get chat history for context
    chat_history = chat_session.get_chat_history()
    
    # Format user input with chat history context
    context = "\n".join([str(message) for message in chat_history])
    enhanced_query = f"Context from previous conversation:\n{context}\n\nCurrent question: {user_input}"
    
    # Format the query for embedding
    userQueryBody = json.dumps({
        "inputText": enhanced_query
    })

    # Get the embedding from the model
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name=region,
    )
    
    # Call Titan Embedding Model
    response = bedrock.invoke_model(
        modelId="amazon.titan-embed-text-v1",
        contentType="application/json",
        body=userQueryBody
    )
    
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')

    # Create OpenSearch client
    awsauth = get_awsauth(region, 'es')
    opensearch = OpenSearch(
        hosts=[{'host': host, 'port': 443}],
        http_auth=awsauth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection
    )

    # Perform k-NN search
    query = {
        "size": 3,  # Number of results to return
        "query": {
            "knn": {
                "embedding_vector": {
                    "vector": embedding,
                    "k": 3
                }
            }
        }
    }

    response = opensearch.search(
        body=query,
        index=index
    )

    # Extract contexts from search results
    contexts = []
    for hit in response['hits']['hits']:
        contexts.append(hit['_source']['text'])

    # Combine contexts
    combined_context = "\n".join(contexts)

    # Create prompt with context and user question
    prompt_data = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": f"Context: {combined_context}\n\nQuestion: {user_input}\n\nProvide a helpful answer based on the context provided. If the context doesn't contain relevant information, say so."
            }
        ],
        "temperature": 0.1,
        "top_p": 0.9,
    }

    # Generate response using Bedrock
    response = bedrock.invoke_model(
        modelId="anthropic.claude-v2",
        contentType="application/json",
        body=json.dumps(prompt_data)
    )
    
    response_body = json.loads(response.get('body').read())
    ai_response = response_body.get('completion')

    # Save the interaction to chat memory
    chat_session.save_message(user_input, ai_response)
    
    return ai_response

def load_chat_history(filename="chat_history.json"):
    """Load chat history from a file"""
    try:
        with open(filename, "r") as f:
            messages_dict = json.load(f)
        chat_session = ChatSession()
        chat_session.memory.chat_memory.messages = messages_from_dict(messages_dict)
        return chat_session
    except FileNotFoundError:
        return ChatSession()

# Example usage
if __name__ == "__main__":
    try:
        load_dotStreat_sl()
        # Initialize or load chat session
        chat_history_file = "chat_history.json"
        chat_session = load_chat_history(chat_history_file)
        
        # OpenSearch configuration
        host = os.environ.get('OPENSEARCH_HOST')
        if not host:
            raise ValueError("OPENSEARCH_HOST environment variable is not set")
        
        region = os.environ.get('AWS_REGION', 'us-east-1')
        print(f"Connected to OpenSearch host: {host}")
        print(f"Using AWS region: {region}")
        print("\nChat session initialized. Type 'exit' to quit, 'clear' to clear chat history.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                # Handle empty input
                if not user_input:
                    print("Please enter a question or type 'exit' to quit.")
                    continue
                
                # Handle exit command
                if user_input.lower() == 'exit':
                    print("\nSaving chat history...")
                    if save_chat_history(chat_session, chat_history_file):
                        print("Chat history saved successfully.")
                    else:
                        print("Failed to save chat history.")
                    print("Goodbye!")
                    break
                
                # Handle clear command
                if user_input.lower() == 'clear':
                    chat_session.clear_memory()
                    print("Chat history cleared.")
                    continue
                
                # Process the query
                print("Processing your question...")
                response = answer_query(user_input, chat_session, host, region)
                
                if response:
                    print("\nAssistant:", response)
                else:
                    print("\nNo response generated. Please try again.")
                
                # Automatically save chat history after each interaction
                save_chat_history(chat_session, chat_history_file)
                
            except KeyboardInterrupt:
                print("\n\nInterrupted by user. Saving chat history...")
                save_chat_history(chat_session, chat_history_file)
                print("Goodbye!")
                break
                
            except Exception as e:
                print(f"\nAn error occurred: {str(e)}")
                print("You can try again or type 'exit' to quit.")
                continue
    
    except Exception as e:
        print(f"Fatal error during initialization: {str(e)}")
        exit(1)

