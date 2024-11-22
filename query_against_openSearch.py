import boto3
import json
from dotenv import load_dotenv
import os
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import streamlit as st

import toml
from pathlib import Path

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

def load_dotStreat():
    """
    Load environment variables from .streamlit/secrets.toml file into os.environ.
    Similar to load_dotenv but for Streamlit secrets.
    
    Returns:
        bool: True if secrets.toml was loaded successfully, False otherwise
    """
    try:
        # Define the path to secrets.toml
        secrets_path = Path('.streamlit/secrets.toml')
        
        # Check if the file exists
        if not secrets_path.exists():
            print(f"Warning: {secrets_path} not found")
            return False
            
        # Load the TOML file
        secrets = toml.load(secrets_path)
        
        # Add each secret to environment variables
        for key, value in secrets.items():
            # Convert all values to strings (environment variables must be strings)
            os.environ[key] = str(value)
            
        return True
        
    except Exception as e:
        print(f"Error loading secrets.toml: {str(e)}")
        return False


def load_dotstream():
    # Get AWS credentials from Streamlit secrets
    aws_credentials = {
        "AWS_ACCESS_KEY_ID": st.secrets["AWS_ACCESS_KEY_ID"],
        "AWS_SECRET_ACCESS_KEY": st.secrets["AWS_SECRET_ACCESS_KEY"],
        "AWS_DEFAULT_REGION": st.secrets["AWS_DEFAULT_REGION"]
    }

    # Initialize Bedrock client
    bedrock_runtime = boto3.client(
        service_name='bedrock-runtime',
        region_name=aws_credentials["AWS_DEFAULT_REGION"],
        aws_access_key_id=aws_credentials["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=aws_credentials["AWS_SECRET_ACCESS_KEY"]
    )
    return bedrock_runtime


# loading in variables from .env file
#load_dotenv()
load_dotStreat_sl()
#bedrock = load_dotstream()
accept = 'application/json'
contentType = 'application/json'
#os.environ['profile_name'] = st.secrets["AWS_PROFILE_NAME"]
#os.environ['opensearch_host'] = st.secrets["AWS_PROFILE_NAME"]
#os.environ['vector_index_name'] = st.secrets["vector_index_name"]
#os.environ['vector_field_name'] = st.secrets["vector_field_name"]

# instantiating the Bedrock client, and passing in the CLI profile
#boto3.setup_default_session(profile_name=os.getenv('profile_name'))
session = boto3.Session(
    aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
    region_name=st.secrets["AWS_DEFAULT_REGION"]
)

bedrock = session.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')

# instantiating the OpenSearch client, and passing in the CLI profile
opensearch = boto3.client("opensearchserverless")
host = os.getenv('opensearch_host')  # cluster endpoint, for example: my-test-domain.us-east-1.aoss.amazonaws.com
region = 'us-east-1'
service = 'aoss'
#credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
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
      

def get_embedding2(body):
    """
    This function is used to generate the embeddings for each question the user submits.
    :param body: This is the question that is passed in to generate an embedding
    :return: A vector containing the embeddings of the passed in content
    """
    # defining the embeddings model
    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    # invoking the embedding model
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    # reading in the specific embedding
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

def amazon(model_id, question):
    """
    Method to invoke Amazon models.
    Args:
        model_id: The specific Amazon model to invoke.
        question: The question passed in by the user on the front end.

    Returns:
        The output text (response) from the specific Amazon model.
    """
    # Define the request body for the Amazon model, passing in the user question
    request_body = json.dumps({"inputText": question,
                                "textGenerationConfig": {
                                    "maxTokenCount": 2000,
                                    "stopSequences": [],
                                    "temperature": 0.5,
                                    "topP": 0.5
                                }})
    # Invoke the Amazon model with the request body, and specific Amazon Model ID selected by the end user
    response = client.invoke_model(
        modelId=model_id,
        body=request_body,
        accept=accept,
        contentType=contentType
    )
    # Extract information from the response
    response_body = json.loads(response.get('body').read())
    # Extract the output text from the response
    output_text = response_body['results'][0]['outputText']
    # Return the output text
    return output_text


def answer_query(user_input):
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
    # Configuring the Prompt for the LLM
    # TODO: EDIT THIS PROMPT TO OPTIMIZE FOR YOUR USE CASE
    prompt_data = f"""\n\nHuman: You are an AI assistant that will help people answer questions they have about [YOUR TOPIC]. Answer the provided question to the best of your ability using the information provided in the Context. 
    Summarize the answer and provide sources to where the relevant information can be found. 
    Include this at the end of the response.
    Provide information based on the context provided.
    Format the output in human readable format - use paragraphs and bullet lists when applicable
    Answer in detail with no preamble
    If you are unable to answer accurately, please say so.
    Please mention the sources of where the answers came from by referring to page numbers, specific books and chapters!

    Question: {userQuery}

    Here is the text you should use as context: {similaritysearchResponse}

    \n\nAssistant:

    """
    # Configuring the model parameters, preparing for inference
    # TODO: TUNE THESE PARAMETERS TO OPTIMIZE FOR YOUR USE CASE
    request_body = json.dumps({"inputText": prompt_data,
                                "textGenerationConfig": {
                                    "maxTokenCount": 2000,
                                    "stopSequences": [],
                                    "temperature": 0.5,
                                    "topP": 0.5
                                }})
    # Run infernce on the LLM

    model_id = "amazon.titan-text-premier-v1:0"  # change this to use a different version from the model provider
    response = bedrock.invoke_model(
        modelId=model_id,
        body=request_body,
        accept=accept,
        contentType=contentType
    )

    # Extract information from the response
    response_body = json.loads(response.get('body').read())
    # Extract the output text from the response
    output_text = response_body['results'][0]['outputText']
    
    return output_text


#question='what is ena?'
#response = answer_query(question)
#print(response)

#response = amazon(model_id= "amazon.titan-text-premier-v1:0" , question=question)
#print(response)