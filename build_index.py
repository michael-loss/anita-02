from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
import os
import json
from langchain.document_loaders import PyPDFDirectoryLoader, PyPDFLoader
from langchain.embeddings import BedrockEmbeddings
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from dotenv import load_dotenv

def load_pdfs_from_folder(folder_path):
    # List to store all documents
    all_documents = []
    
    # Iterate through files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith('.pdf'):
            try:
                # Create full file path
                file_path = os.path.join(folder_path, filename)
                
                # Load PDF
                loader = PyPDFLoader(file_path)
                documents = loader.load()
                
                # Add to our documents list
                all_documents.extend(documents)
                print(f"Successfully loaded: {filename}")
                
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    
    return all_documents

def delete_index(client, index_name):
    if client.indices.exists(index_name):
        client.indices.delete(index=index_name)
        print(f"Deleted existing index: {index_name}")


"""Create index with vector mapping"""
def create_vector_index(client,index_name,vector_dimension):
    # Define the index mapping
    index_mapping = {
        "mappings": {
            "properties": {
                "vectors": {
                    "type": "knn_vector",
                    "dimension": vector_dimension,
                    "method": {
                        "name": "hnsw",
                        "space_type": "l2",
                        "engine": "nmslib",
                        "parameters": {
                            "ef_construction": 128,
                            "m": 16
                        }
                    }
                },
                "text": {"type": "text"},
                "metadata": {"type": "object"}
            }
        },
        "settings": {
            "index": {
                "knn": True,
                "knn.algo_param.ef_search": 100
            }
        }
    }

    # Create the index if it doesn't exist
    if not client.indices.exists(index_name):
        client.indices.create(
            index=index_name,
            body=index_mapping
        )
        print(f"Created index: {index_name}")
    else:
        print(f"Index {index_name} already exists")

# Initialize the OpenSearch client

def init_opensearch_client():
    # Get credentials
    credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
    auth = AWSV4SignerAuth(credentials, 'us-east-1', 'aoss')
    
    # Initialize the OpenSearch client
    client = OpenSearch(
        hosts=[{'host': os.getenv('opensearch_host'), 'port': 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=300
    )
    
    return client

# Example usage:
try:
    client = init_opensearch_client()
    print("Successfully connected to OpenSearch")
except Exception as e:
    print(f"Error connecting to OpenSearch: {str(e)}")


def main():

    # Load environment variables
    load_dotenv()

    # Initialize client
    client = init_opensearch_client()
    
    # Create index with proper mapping
    create_vector_index(
        client,
        index_name=os.getenv('vector_index_name'),
        vector_dimension=1536  # Adjust based on your embedding model
    )

def setup_vector_store():
    credentials = boto3.Session(profile_name=os.getenv('profile_name')).get_credentials()
    auth = AWSV4SignerAuth(credentials, 'us-east-1', 'aoss')
    # Initialize embeddings
    embeddings = BedrockEmbeddings(
        credentials_profile_name=os.getenv('profile_name'),
        model_id="amazon.titan-embed-text-v1"
    )
    
    # Initialize vector store
    vector_store = OpenSearchVectorSearch(
        index_name=os.getenv('vector_index_name'),
        embedding_function=embeddings,
        opensearch_url=f"https://{os.getenv('opensearch_host')}:443",
        vector_field="vectors",  # Make sure this matches the mapping
        is_aoss=True,
        http_auth=auth  # Use the same auth as above
    )
    
    return vector_store

def process_documents(documents, vector_store):
    try:
        vector_store.add_documents(documents)
        print("Successfully added documents to vector store")
    except Exception as e:
        print(f"Error adding documents: {str(e)}")

def get_embedding(body):
    """
    This function is used to generate the embeddings for a specific chunk of text
    :param body: This is the example content passed in to generate an embedding
    :return: A vector containing the embeddings of the passed in content
    """
    bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')

    modelId = 'amazon.titan-embed-text-v1'
    accept = 'application/json'
    contentType = 'application/json'
    response = bedrock.invoke_model(body=body, modelId=modelId, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding

def indexDoc(client, vectors, text):
    """
    This function indexing the documents and vectors into Amazon OpenSearch Serverless.
    :param client: The instatiation of your OpenSearch Serverless instance.
    :param vectors: The vector you generated with the get_embeddings function that is going to be indexed.
    :param text: The actual text of the document you are storing along with the vector of that text.
    :return: The confirmation that the document was indexed successfully.
    """
    # TODO: You can add more metadata fields if you wanted to!
    indexDocument = {
        os.getenv("vector_field_name"): vectors,
        'text': text

    }
    # Configuring the specific index
    response = client.index(
        index=os.getenv("vector_index_name"),
        body=indexDocument,
        refresh=False
    )
    print(response)
    return response

def populate_main():

    boto3.setup_default_session(profile_name=os.getenv('profile_name'))
    #bedrock = boto3.client('bedrock-runtime', 'us-east-1', endpoint_url='https://bedrock-runtime.us-east-1.amazonaws.com')
    opensearch = boto3.client("opensearchserverless")
    # Create index first
    client = init_opensearch_client()
    vector_dimension=1536
    index_name = os.getenv('vector_index_name')
    delete_index(client, index_name)

    create_vector_index(client, os.getenv('vector_index_name'),vector_dimension)
    
    # Then set up vector store
    vector_store = setup_vector_store()
    pdf_path = "/Users/TheDr/Desktop/position-statements"

    documents = load_pdfs_from_folder(pdf_path)

    text_splitter = RecursiveCharacterTextSplitter(
    # Play with Chunk Size
    chunk_size=600,
    chunk_overlap=100,
    )


    doc = text_splitter.split_documents(documents)


    avg_doc_length = lambda documents: sum([len(doc.page_content) for doc in documents]) // len(documents)
    avg_char_count_pre = avg_doc_length(documents)
    avg_char_count_post = avg_doc_length(doc)
    print(f'Average length among {len(documents)} documents loaded is {avg_char_count_pre} characters.')
    print(f'After the split we have {len(doc)} documents more than the original {len(documents)}.')
    print(f'Average length among {len(doc)} documents (after split) is {avg_char_count_post} characters.')
    
    for i in doc:
        # The text data of each chunk
        exampleContent = i.page_content
        # Generating the embeddings for each chunk of text data
        exampleInput = json.dumps({"inputText": exampleContent})
        exampleVectors = get_embedding(exampleInput)
        # setting the text data as the text variable, and generated vector to a vector variable
        # TODO: You can add more metadata fields here if you wanted to by configuring it here and adding them to the indexDocument dictionary above
        text = exampleContent
        vectors = exampleVectors
        # calling the indexDoc function, passing in the OpenSearch Client, the created vector, and corresponding text data
        # TODO: If you wanted to add metadata you would pass it in here
        indexDoc(client, vectors, text)
        
    
    #folder_path ='/Users/TheDr/Desktop/position-statements/Access to Quality Healthcare Position Statement.pdf'
    # Load and process documents
    #loader = PyPDFDirectoryLoader(folder_path)
    #documents = loader.load()
    #documents =PyPDFDirectoryLoader(folder_path)
    #loader = PyPDFLoader(folder_path)
    #documents = loader.load()
    #process_documents(documents, vector_store)


if __name__ == "__main__":
    populate_main()
