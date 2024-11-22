import boto3
import json
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import streamlit as st
import toml
from pathlib import Path
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
import time
#from langchain.memory import ChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from requests_aws4auth import AWS4Auth


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

def get_awsauth(region, service):
    credentials = boto3.Session().get_credentials()
    return AWS4Auth(
        credentials.access_key,
        credentials.secret_key,
        region,
        service,
        session_token=credentials.token
    )

def get_model_info(response, requested_model_id):
    response_body = json.loads(response.get('body').read())
    metadata = response.get('ResponseMetadata', {})
    headers = metadata.get('HTTPHeaders', {})
    
    # Try different ways to get model info
    model_info = {
        'requested_model': requested_model_id,
        'actual_model': None,
        'provider': None
    }
    
    # Try to get from response body (some models include this)
    if 'modelId' in response_body:
        model_info['actual_model'] = response_body['modelId']
    
    # Try to get from headers
    model_header = headers.get('x-amzn-bedrock-model-id')
    if model_header:
        model_info['actual_model'] = model_header
    
    # Extract provider from model ID
    if requested_model_id:
        provider = requested_model_id.split('.')[0]  # e.g., 'meta' from 'meta.llama3-70b-instruct-v1'
        model_info['provider'] = provider

    return model_info

def load_dotStreat_sl():
    """
    Load environment variables from either Streamlit Cloud secrets or local .streamlit/secrets.toml
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
                print(f"Warning: {secrets_path} not found")
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
        print(f"Error loading secrets: {str(e)}")
        return False

# Initialize AWS session and clients
load_dotStreat_sl()

session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

bedrock = session.client('bedrock-runtime', 'us-east-1', 
                        endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')

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

def get_embedding(text):
    """
    Get embeddings using Amazon Titan embedding model
    """
    # Create the Bedrock client
    bedrock = boto3.client(
        service_name='bedrock-runtime',
        region_name='us-east-1'
    )
    
    # Prepare the request
    request_body = {
        "inputText": text
    }
    
    # Invoke the model
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )
    
    # Process the response
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    
    return embedding

def get_embedding2(text):
    """Get embeddings using Amazon Titan embedding model"""
    if isinstance(text, str):
        request_body = {"inputText": text}
    else:
        request_body = text
        
    response = bedrock.invoke_model(
        modelId='amazon.titan-embed-text-v1',
        body=json.dumps(request_body),
        contentType='application/json',
        accept='application/json'
    )
    
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding



def answer_query_titan(user_input, chat_handler):
    """
    This function takes the user question, creates an embedding of that question,
    and performs a KNN search on your Amazon OpenSearch Index. Using the most similar results it feeds that into the Prompt
    and LLM as context to generate an answer.
    :param user_input: This is the natural language question that is passed in through the app.py file.
    :return: The answer to your question from the LLM based on the context that was provided by the KNN search of OpenSearch.
    """
    # Setting primary variables, of the user input
    userQuery = user_input
    # formatting the user input
    userQueryBody = json.dumps({"inputText": userQuery})
    # creating an embedding of the user input to perform a KNN search with
    userVectors = get_embedding(userQueryBody)
    # the query parameters for the KNN search performed by Amazon OpenSearch with the generated User Vector passed in.
    # TODO: If you wanted to add pre-filtering on the query you could by editing this query!
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
    # performing the search on OpenSearch passing in the query parameters constructed above
    response = client.search(
        body=query,
        index=st.secrets["vector_index_name"]#os.getenv("vector_index_name")
    )

    # Format Json responses into text
    similaritysearchResponse = ""
    # iterating through all the findings of Amazon openSearch and adding them to a single string to pass in as context
    for i in response["hits"]["hits"]:
        outputtext = i["fields"]["text"]
        similaritysearchResponse = similaritysearchResponse + "Info = " + str(outputtext)

        similaritysearchResponse = similaritysearchResponse

    #chat history
    chat_history = chat_handler.get_conversation_string()  

    # Configuring the Prompt for the LLM
    # TODO: EDIT THIS PROMPT TO OPTIMIZE FOR YOUR USE CASE
    
    prompt_data = f"""\n\nAssistant: You are an AI assistant that will help members of the Emergency Nurses Association (ENA) find information about ENA's position statements. Answer the provided question to the best of your ability using the information provided in the Context.

Summarize the answer and provide sources to where the relevant information can be found, including links to ENA's website, position statement documents, and relevant policy briefs.

Include this at the end of the response.
Provide information based on the context provided.
Format the output in a human-readable format - use paragraphs and bullet lists when applicable.
Answer in detail with no preamble.
If you are unable to answer accurately, please say so.
Please mention the sources of where the answers came from by referring to specific ENA documents, policy briefs, and webpage URLs.

Previous conversation: {chat_history}

Question: {userQuery}

Here is the text you should use as context: {similaritysearchResponse}

\n\nAssistant:

    """
    #Configuring the model parameters, preparing for inference
    #TODO: TUNE THESE PARAMETERS TO OPTIMIZE FOR YOUR USE CASE
    request_body = json.dumps({"inputText": prompt_data,
                                "textGenerationConfig": {
                                    "maxTokenCount": 2000,
                                    "stopSequences": [],
                                    "temperature": .1,
                                    "topP": 0.3
                                }})
    # Run infernce on the LLM

    model_id = "amazon.titan-text-premier-v1:0"  # change this to use a different version from the model provider
    response = bedrock.invoke_model(
        modelId=model_id,
        body=request_body,
        accept=accept,
        contentType=contentType
    )

    # Get model information
    #model_info = get_model_info(response, model_id)


    # Extract information from the response
    response_body = json.loads(response.get('body').read())
    # Extract the output text from the response
    output_text = response_body['results'][0]['outputText']
    

    # Save interaction to chat memory
    chat_handler.add_message("human", user_input)
    chat_handler.add_message("ai", output_text)

    output_text = f"{output_text}\n\nModel used: {model_id}"
  
    return output_text

def answer_query_llama(user_input, chat_handler):
    """
    This function takes the user question, creates an embedding of that question,
    and performs a KNN search on your Amazon OpenSearch Index. Using the most similar results it feeds that into the Prompt
    and LLM as context to generate an answer.
    :param user_input: This is the natural language question that is passed in through the app.py file.
    :return: The answer to your question from the LLM based on the context that was provided by the KNN search of OpenSearch.
    """
    # Setting primary variables, of the user input
    userQuery = user_input
    # formatting the user input
    userQueryBody = json.dumps({"inputText": userQuery})
    # creating an embedding of the user input to perform a KNN search with
    userVectors = get_embedding(userQueryBody)
    # the query parameters for the KNN search performed by Amazon OpenSearch with the generated User Vector passed in.
    # TODO: If you wanted to add pre-filtering on the query you could by editing this query!
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
    # performing the search on OpenSearch passing in the query parameters constructed above
    response = client.search(
        body=query,
        index=st.secrets["vector_index_name"]#os.getenv("vector_index_name")
    )

    # Format Json responses into text
    similaritysearchResponse = ""
    # iterating through all the findings of Amazon openSearch and adding them to a single string to pass in as context
    for i in response["hits"]["hits"]:
        outputtext = i["fields"]["text"]
        similaritysearchResponse = similaritysearchResponse + "Info = " + str(outputtext)

        similaritysearchResponse = similaritysearchResponse

    #chat history
    chat_history = chat_handler.get_conversation_string()  

    # Configuring the Prompt for the LLM
    # TODO: EDIT THIS PROMPT TO OPTIMIZE FOR YOUR USE CASE
    
    prompt_data = f"""\n\nAssistant: You are an AI assistant that will help members of the Emergency Nurses Association (ENA) find information about ENA's position statements. Answer the provided question to the best of your ability using the information provided in the Context.

Summarize the answer and provide sources to where the relevant information can be found, including links to ENA's website, position statement documents, and relevant policy briefs.

Include this at the end of the response.
Provide information based on the context provided.
Format the output in a human-readable format - use paragraphs and bullet lists when applicable.
Answer in detail with no preamble.
If you are unable to answer accurately, please say so.
Please mention the sources of where the answers came from by referring to specific ENA documents, policy briefs, and webpage URLs.
End by printing the model_id and the name of the model used for this response.

Previous conversation: {chat_history}

Question: {userQuery}

Here is the text you should use as context: {similaritysearchResponse}

\n\nAssistant:

    """

    formatted_prompt = f"""
        <|begin_of_text|><|start_header_id|>user<|end_header_id|>
        {prompt_data}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """

# TODO: TUNE THESE PARAMETERS TO OPTIMIZE FOR YOUR USE CASE
    request_body = json.dumps({
    "prompt": formatted_prompt,  # Changed from "input" to "prompt"
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

    # Extract information from the response
    response_body = json.loads(response.get('body').read())
    # For Llama 3, the response structure is different
    output_text = response_body.get('generation', '')  # Use 'generation' instead of accessing ['results'][0]['outputText']

    # Save interaction to chat memory
    chat_handler.add_message("human", user_input)
    chat_handler.add_message("ai", output_text)

    output_text = f"{output_text}\n\nModel used: {model_id}"

    return output_text

import streamlit as st

# Define your main function
def main():
    # Create a sidebar for the left panel
    with st.sidebar:
        # Display the image at the top of the left panel
        col1, col2 = st.columns([0.2, 0.8])  # Adjust the column width to position items side by side
        with col1:
            st.image("AnitaMDorr.jpg", width=200)  # Adjust width as needed
        with col2:
            st.title("Hello! I'm ANITA")

        # Add radio button group for "ENA Focus"
        enafocus = st.radio(
            "ENA Focus",
            ("Position Statements", "Education"),
            index=0,  # Default to "Position Statements"
            help="Select the ENA focus area"
        )

        # Add radio button group for "LLM Model"
        llm_model = st.radio(
            "LLM Model",
            ("Llama", "Titan"),
            index=1,  # Default to "Titan"
            help="Select the LLM model"
        )

        # Add clear chat button in the sidebar
        clear_button = st.button("🧹", help="Clear conversation")
        if clear_button:
            st.session_state.chat_handler = ChatHandler()
            st.rerun()

    # Set the prompt based on ENA Focus selection
    if enafocus == "Position Statements":
        chat_input_prompt = "Ask me anything about ENA's position statements!"
    elif enafocus == "Education":
        chat_input_prompt = "Ask me anything about ENA's Education offerings!"

    # Set the response function based on LLM Model selection
    if llm_model == "Titan":
        response_function = answer_query_titan
    elif llm_model == "Llama":
        response_function = answer_query_llama

    # Create a container for the header with subtitle (this will be the main content area)
    header_container = st.container()
    # with header_container:
    #     col1, col2 = st.columns([0.9, 0.1])
    #     with col1:
    #         st.write("ENA's position statements")

    # Add custom CSS to style the button and layout
    st.markdown("""
        <style>
        .stButton>button {
            background-color: transparent;
            border: none;
            color: #4F8BF9;
            margin-top: 0px;
            padding: 5px 10px;
            border-radius: 5px;
        }
        .stButton>button:hover {
            background-color: #f0f2f6;
            color: #4F8BF9;
        }
        </style>
        """, unsafe_allow_html=True)

    # Initialize chat handler in session state if not already initialized
    if 'chat_handler' not in st.session_state:
        st.session_state.chat_handler = ChatHandler()

    # Display chat history
    for message in st.session_state.chat_handler.get_chat_history():
        with st.chat_message(message.type):
            st.write(message.content)

    # Get user input
    prompt = st.chat_input(chat_input_prompt)
    if prompt:
        # Display user message
        with st.chat_message("human"):
            st.write(prompt)

        # Get and display AI response based on the selected model
        with st.chat_message("ai"):
            with st.spinner("Thinking..."):
                response = response_function(prompt, st.session_state.chat_handler)
                st.write(response)

# Call the main function to run the app
if __name__ == "__main__":
    main()
