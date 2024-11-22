import os
import json
import boto3
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
from langchain_community.vectorstores import OpenSearchVectorSearch
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
from query_against_openSearch import load_dotStreat_sl

# Load environment variables
load_dotStreat_sl()

def initialize_clients():
    try:
        # Initialize Bedrock client
        bedrock_runtime = boto3.Session(
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_DEFAULT_REGION"]
        ).client('bedrock-runtime')

        # Get OpenSearch host
        host = os.getenv('opensearch_host')
        if not host:
            raise ValueError("OpenSearch host not found in environment variables")

        # Create AWS credentials
        credentials = boto3.Session(
            aws_access_key_id=st.secrets["AWS_ACCESS_KEY_ID"],
            aws_secret_access_key=st.secrets["AWS_SECRET_ACCESS_KEY"],
            region_name=st.secrets["AWS_DEFAULT_REGION"]
        ).get_credentials()

        # Create AWS auth
        awsauth = AWS4Auth(
            credentials.access_key,
            credentials.secret_key,
            st.secrets["AWS_DEFAULT_REGION"],
            'aoss',
            session_token=credentials.token if hasattr(credentials, 'token') else None
        )

        # Initialize OpenSearch client
        opensearch_client = OpenSearch(
            hosts=[{'host': host, 'port': 443}],
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )

        return bedrock_runtime, opensearch_client, host, awsauth
    except Exception as e:
        st.error(f"Error initializing clients: {str(e)}")
        raise

def create_index_with_mapping(client, index_name):
    index_body = {
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 512
            }
        },
        "mappings": {
            "properties": {
                "vector_field": {
                    "type": "knn_vector",
                    "dimension": 1536,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 512,
                            "m": 16
                        }
                    }
                },
                "text": {
                    "type": "text"
                },
                "metadata": {
                    "type": "object"
                }
            }
        }
    }
    
    try:
        # Delete index if it exists
        if client.indices.exists(index=index_name):
            client.indices.delete(index=index_name)
            print(f"Existing index {index_name} deleted")

        # Create new index
        client.indices.create(index=index_name, body=index_body)
        print(f"Index {index_name} created successfully")
    except Exception as e:
        print(f"Error managing index: {e}")
        raise

def initialize_bedrock_llm(bedrock_runtime):
    try:
        return Bedrock(
            model_id="amazon.titan-text-premier-v1:0",
            client=bedrock_runtime,
            model_kwargs={
                "maxTokenCount": 4096,
                "temperature": 0.7,
                "topP": 0.9,
            }
        )
    except Exception as e:
        st.error(f"Error initializing Bedrock LLM: {str(e)}")
        raise

def initialize_embeddings(bedrock_runtime):
    try:
        return BedrockEmbeddings(
            client=bedrock_runtime,
            model_id="amazon.titan-embed-text-v1"
        )
    except Exception as e:
        st.error(f"Error initializing embeddings: {str(e)}")
        raise

def initialize_vector_store(embeddings, host, awsauth, index_name):
    try:
        return OpenSearchVectorSearch(
            index_name=index_name,
            embedding_function=embeddings,
            opensearch_url=f"https://{host}",
            http_auth=awsauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection,
            vector_field="vector_field",
            engine="nmslib"
        )
    except Exception as e:
        st.error(f"Error initializing vector store: {str(e)}")
        raise

def initialize_conversation_chain(llm, vector_store):
    try:
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        return ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vector_store.as_retriever(search_kwargs={'k': 3}),
            memory=memory,
            return_source_documents=True
        )
    except Exception as e:
        st.error(f"Error initializing conversation chain: {str(e)}")
        raise

def create_streamlit_interface():
    try:
        st.title("RAG Chatbot with Amazon Bedrock")
        
        # Initialize session state
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Initialize all clients first and capture the returned values
        bedrock_runtime, opensearch_client, host, awsauth = initialize_clients()
        
        # Create/update index
        create_index_with_mapping(opensearch_client, st.secrets["vector_index_name"])
        
        # Initialize components
        llm = initialize_bedrock_llm(bedrock_runtime)
        embeddings = initialize_embeddings(bedrock_runtime)
        vector_store = initialize_vector_store(
            embeddings, 
            host, 
            awsauth, 
            st.secrets["vector_index_name"]
        )
        conversation_chain = initialize_conversation_chain(llm, vector_store)
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("What would you like to know?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                response_container = st.empty()
                try:
                    response = conversation_chain({"question": prompt})
                    answer = response['answer']
                    response_container.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except Exception as e:
                    error_message = f"An error occurred: {str(e)}"
                    response_container.error(error_message)
                    st.error(error_message)

    except Exception as e:
        st.error(f"Application Error: {str(e)}")

if __name__ == "__main__":
    create_streamlit_interface()
