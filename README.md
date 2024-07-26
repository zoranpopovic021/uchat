# uchat v0.5 2024

AI chatbot based on django framework and Retrieal Augmented Generation with state machine filtering
and multuple BCS languages (Bosnian/Croatian/Serbian/Macedonian etc)
Demo vector database (ChromaDB) created by importing PDF documents on internet for EON devices,
and examples for helpful chat with document contents.
# To Do:
        MHA E/D network for spelling and declination errors, based on logged discussions as an inbuilt feature,
        sentiment analysis for improved state propagation.
        UI options to delete database, beside existing CRUD opeations.

# How to start it:

Prerequisite: pytho3n 3.11 installed, with OpenAI, ChromaDB and other packages.

Pull the main branch with:

 git clone https://git.hub/zoranpopovic021/uchat --branch main

Use pip3 to install missing packages. LLM and embeddings downloads will be handled by HuggingFace.

Unpack demo vector DB in chroma_data.tgz first, and then start WSGI on local host/port with:

  python3 manage.py runserver localhost:8080
  
LLM in this demo is based on Yugo60-GPT by datatabb, but this can be customized in the app UI, too.

