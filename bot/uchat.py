import chromadb
from chromadb.utils import embedding_functions
from .models import Param
import os
import torch
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from itertools import islice, zip_longest
import re

# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
model_id = "Yugo60-GPT-GGUF.Q4_K_M.gguf"

#outputs = model.generate(**inputs, max_new_tokens=20)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if not torch.backends.mps.is_available():
    if not torch.backends.mps.is_built():
        print("MPS not available because the current PyTorch install was not "
              "built with MPS enabled.")
    else:
        print("MPS not available because the current MacOS version is not 12.3+ "
              "and/or you do not have an MPS-enabled device on this machine.")
else:
    torch.set_default_device("mps")

model = ""
CHROMA_DATA_PATH = "/Users/zoranpopovic/uchat/chroma_data/"
EMBED_MODEL = "all-MiniLM-L6-v2"
# NousResearch/Hermes-2-Pro-Mistral-7B 
# distilbert-base-multilingual-case
# paraphrase-multilingual-MiniLM-L12-v2d
COLLECTION_NAME = "chroma_data"
PDF_PATH = "./PDF/uputstvo_uz_eon_smart_box-1.pdf"
PDF_PATH2 = "./PDF/uputstvo_uz_eon_smart_aplikaciju-1.pdf" 
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50
max_results = 3 
min_len = 40
min_distance = 0.35
max_distance = 0.6
temperature = 0.55
max_tokens=3072
top_p=0.8
frequency_penalty=0.0
presence_penalty=0.15
DEBUG = True
system_sr = "Zoveš se U-Chat AI asistent i pomažeš korisniku usluga kompanije United Group. Korisnik postavlja pitanje ili problem, upareno sa dodatnima saznanjima. Na osnovu toga napiši korisniku kratak i ljubazan odgovor koji kompletira njegov zahtev ili mu daje odgovor na pitanje. "
# " Ako ne znaš odgovor, reci da ne znaš, ne izmišljaj ga."
system_sr += "Usluge kompanije United Group uključuju i kablovsku mrežu za digitalnu televiziju, pristup internetu, uređaj EON SMART BOX za TV sadržaj, kao i fiksnu telefoniju."
system = {'srpski': system_sr, 'hrvatski': "", 'slovenački': "", 'makedonski': ""}
ctxpre = ""
msg_content = {'srpski': "- Dodatna saznanja su: ", 'hrvatski': "", 'slovenački': "", 'makedonski': ""}
max_conv = 3
try:
   edit_all = Param.objects.all()
   for edit in edit_all:
      system[edit.jezik] = edit.system
      ctxpre = edit.ctxpre
      msg_content[edit.jezik] = edit.msg_content
      min_len = edit.min_len
      CHUNK_SIZE = edit.CHUNK_SIZE
      CHUNK_OVERLAP = edit.CHUNK_OVERLAP
      max_results = edit.max_results
      EMBED_MODEL = edit.EMBED_MODEL
      model_id = edit.model_id
      min_distance = edit.min_distance
      max_distance = edit.max_distance
      max_conv = edit.max_conv
      temperature = edit.temperature
      top_p = edit.top_p
      max_tokens = edit.max_tokens
      presence_penalty = edit.presence_penalty
      frequency_penalty = edit.frequency_penalty
      DEBUG = edit.DEBUG
except:
   pass

def load_and_split_document(pdf_path):
    loader = PyPDFLoader(pdf_path)
    print('Loaded: ' + pdf_path)
    return loader.load_and_split()

def split_text_into_chunks(pages, chunk_size, chunk_overlap):
    n = -1
    for page in range(len(pages)): pages[page].page_content = re.sub(r'\s+'," ", pages[page].page_content.replace(". .","").replace(r'\n','.')).replace('..','')
    for p in range(len(pages)):
      if len(pages[p].page_content)<min_len:
         if n<0: n = p
      else:
         if n>=0:
            pages[n]=pages[p]; n += 1
    if n>0: pages = pages[:n-1]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(pages)

def batched(iterable, n):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch

#client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
#client.allow_reset = True
#client.delete_collection(COLLECTION_NAME)
oc = OpenAI(base_url="http://localhost:4891/v1", api_key="not-needed")

chroma_client = chromadb.PersistentClient(CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

collection = chroma_client.get_or_create_collection(
        name="chroma_data",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

last = collection.count()

def update_collection(docs, last, jezik): 
   state = -2
   used =[]
   for g in docs[0::2]:
      state += 2
      documents=docs[state+1][0]
      bot.uchat.collection.add(
        documents=documents,
        ids=[f"id{last+i}" for i in range(len(documents))],
        metadatas=[{"state": g, "next": g, "used": False, "source": 'None', "page": -1, "lang": jezik } for i in range(len(documents)) ]
      )
      last += len(documents)
      if (len(docs[state+1])>1):
         for n in docs[state+1][1:]:
            bot.uchat.collection.add(
    	    documents=n[1:],
	     ids=[f"id{last+i-1}" for i in range(1,len(n))],
              metadatas=[{"state": g, "next": n[0], "used": False, "source": 'None', "page": -1, "lang": jezik } for i in range(1,len(n)) ]
            )
         for i in range(1,len(n)): used += [0]
         last += len(n)-1
   return last

#docus = load_and_split_document(PDF_PATH) + load_and_split_document(PDF_PATH2)

def load_docs(path, jezik):
   docus = load_and_split_document(path)
   pages = split_text_into_chunks(docus, CHUNK_SIZE, CHUNK_OVERLAP)
   document_indices = list(range(bot.uchat.last, bot.uchat.last+len(pages)))
   for batch in batched(document_indices, 66):
        bot.uchat.collection.add(
            ids=[f"id{last+batch[i]}" for i in range(len(batch))],
            documents=[pages[i].page_content for i in batch],
            metadatas=[dict(dict(dict(dict(pages[i].metadata, used=False), next='None'), state='None'), lang=jezik) for i in batch],
        )
        last += len(batch)
   return last

