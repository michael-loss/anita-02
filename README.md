This code is based on the code provided by aws in https://github.com/aws-samples/genai-quickstart-pocs/tree/main/genai-quickstart-pocs-python/amazon-bedrock-rag-opensearch-serverless-poc

This initial version has been modified as follows:

It uses RAG and opensearch based on ENA position statements provided by daryl
It handles api keys and secret keys using .streamlit/secret.toml to make it easier to post the chatbot as a streamlit app in a way that doesnt expose any sensitive access info doe aws.
It currently uses amazon.titan-text-premier-v1:0 as the primamry llm and amazon.titan-embed-text-v1 for embeddings for prompts.
To get get ur chat bot up and running follow the directions below.

pip install virtualenv 
python3 -m venv venv


The virtual environment will be extremely useful when you begin installing the requirements. If you need more clarification on the creation of the virtual environment please refer to this [blog](https://www.freecodecamp.org/news/how-to-setup-virtual-environments-in-python/).
After the virtual environment is created, ensure that it is activated, following the activation steps of the virtual environment tool you are using. Likely:


***
cd venv
cd bin
source activate
cd ../../
***

After your virtual environment has been created and activated, you can install all the requirements found in the requirements.txt file by running this command in the root of this repos directory in your terminal:

pip install -r requirements.txt

# Anita-br01
