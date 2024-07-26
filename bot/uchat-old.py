import chromadb
from chromadb.utils import embedding_functions
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from itertools import islice, zip_longest
import re

#model_id = "mistralai/Mistral-7B-Instruct-v0.2"
#tokenizer = AutoTokenizer.from_pretrained(model_id)
model_id = "Yugo60-GPT-GGUF.Q4_K_M.gguf"

#inputs = tokenizer(text, return_tensors="pt").to(0)

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
CHROMA_DATA_PATH = "chroma_data/"
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

client = chromadb.PersistentClient(path=CHROMA_DATA_PATH)
client.allow_reset = True
client.delete_collection(COLLECTION_NAME)

oc = OpenAI(base_url="http://localhost:4891/v1", api_key="not-needed")

docs = [
  "s0", [
     [
       "",
       "Navedite, da li je problem sa EON uređajem, upravljačem ili WiFi mrežom?",
       "Problem sa TV odabirom sadržaja, prijemom signala ili dostupnošću usluga?",
       "Problem sa naplatom usluga?",
     ],
     [ "s00", "Ako korisnik ima problema sa prijemom kablovskog signala ili slikom, proveriti EON SMART BOX uređaj", "Ako korisnik ima problem sa EON SMART BOX uređajem ili aplikacijom, treba proveriti EON podešavanja", "Ako je korisniku poptpuno crn ekran, proveriti resetovanje EON SMART BOX uređaja ili aplikacije" ],
     [ "s01", "Ako nee radi internet korisniku, treba proveriti kablovski prijem", "Ako ne radi WiFi korisniku, onda proveriti priključen WiFI ruter" ],
     [ "s02", "Ako korisnik ima problema sa izborom kanala, proveriti podešavanja ili pretplatu", "Ako ne radi daljinski upravljač, proveriti EON SMART BOX uređaj" ],
     [ "s03", "Ako korisnik ima problema s plaćanjem pretplate, neka ga proveri ili izmiri na web stranici ili najbližoj filijali.", "Ako korisnik dobije poruku o neizmirenim obavezama na ekranu, neka proveri stanje pretplate na web strasnici ili u najbližoj filijali.", "Ako korisnik želi proveru stanja bez web stranice ili filijale, neka se obrati Call centru", "Ako korisnik želi naplatu obaveza ili promenu paketa usluga, neka se obrati Call centru." ],
  ],
  "s00", [
     [
       "Ako korisnik nije resetovao EON SMART BOX uređaj ili aplikaciju, neka pokuša da ga restartuje, kao i TV uređaj.",
       "Ako problem korisniku nije rešen resetovanjem, onda ga uputiti na Call centar.",
     ],
     [ "s000", "Ako je resetovanje bilo uspešno, ali korisnik i dalje ima problem, neka se obrati call centru.", "Ako korisnik ne zna kako da resetuje uređaj, neka ga isključi iz struje na nekoliko sekundi, i ponovo uključi." ],
     [ "s99", "Ako je korisniku nejasan odgovor, neka se obrati Call centru."],
  ],
  "s01", [
     [
        "Ako korisnik ima problema sa internetom, neka pokuša da restartuje WiFi ruter.",
        "Ako se korisniku problem nastavlja i nakon resetovanja, neka pokuša sa resetovanje EON SMART BOX i WiFI uređaja.",
     ],
     [ "s010", "Ako korisnik ima problem sa prijemom kablovskog i internet signala, a resetovanje nije pomoglo, neka se obrati Call centru.", "Ako korisnik ima kablovski prijem, ali mu internet ne radi, neka proveri da li radim ping 8.8.8.8"],
     [ "s00", "Ako korisnik ima kablovski prijem, ali internet ne radi, neka resetuje WiFI rutter", "Ako korisnuku rade TV kanali ali ne i internet, neka resetuje WiFI ruter", "Ako korisnik može gledati TV ali mu internet ne radi, neka resetuje WiFi ruter."],
     [ "s99", "Ako je korisniku nejasan odgovor, neka se obrati Call centru." ],
  ],
  "s02", [
     [
        "Ako korisniku nisu dostpni samo pojedini kanali, neka proveri stanje pretplate.",
        "Ako korisniku nije dostupna nijedna funkcija na daljinskom upravljaču, neka pokuša da promeni baterije ili daljinski upravljać.",
     ],
     [ "s03", "Ako korisnik želi da promeni paket usluga, neka se obrati Call centru.", "Ako korisnik želi da proveri stanje pretplate na svom računu, neka proveri stanje na web stranici ili preko Call centra." ],
     [ "s99", "Ako je korisniku nejasan odgovor, neka se obrati Call centru." ],
  ],
  "s03", [
     [
        "Ako niste izmirili Vaše redovne obaveze, možete to učiniti online, ili putem najbližeg predstavništva.",
     ],
     [ "s99", "Ako je korisniku nejasan odgovor, neka se obrati Call centru." ],
  ],
  "s09", [
     [
        "Kontaktirajte Call centar za korisničku podrškua na 0800123123123",
     ],
     [ "s0", "Ako je korisniku nejasan odgovor, neka se obrati Call centru."]
  ],
  "s000", [
    [
        "Ako korisnik ne zna kako da resetuje uređaj ili aplikaciju, neka isključi uređaj iz struje, i nakon nekoliko sekundi ponovo ga uključi.",
    ],
    [ "s99", "Ako korisniku i dalje ima problem, neka se obrati Call centru.", "Ako korisnik ima novi problem, zahtevaj pojašnjenje." ],
  ],
  "s010", [
    [  
        "Ako korisniku radi ping a ima problem sa internetom, neka pokuša da postavi u svom browser-u ili mrežnim podešavanjima korisničkog uređaja 8.8.8.8 kao DNS",
    ],
    [ "s99", "Ako se korisnik žali da mu nešto ne radi i dalje, uputite ga na Call centar", "Ako korisniku ne pomaže odgovor, neka se obrati Call centru." ],
  ],
  "s99", [
     [
        "Odgovorite korisniku na sledeći način isključivo: Žao mi je, ali ne mogu Vam pomoći. Kontaktirajte Call centar na 08001231231223"
     ],
     [ "s0", "Ako je korisniku nejasan odgovor, neka se obrati Call centru." ],
  ]
]

if not os.path.exists(CHROMA_DATA_PATH):
     os.makedirs(CHROMA_DATA_PATH)

chroma_client = chromadb.PersistentClient(CHROMA_DATA_PATH)

embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

collection = chroma_client.get_or_create_collection(
        name="chroma_data",
        embedding_function=embedding_func,
        metadata={"hnsw:space": "cosine"},
    )

#async function main() {
last = 0
state = -2
used =[] 
for g in docs[0::2]:
   state += 2
   documents=docs[state+1][0]
   collection.add(
     documents=documents,
     ids=[f"id{last+i}" for i in range(len(documents))],
     metadatas=[{"state": g, "next": g, "used": False, "source": 'None', "page": -1 } for i in range(len(documents)) ]
   )
   for i in range(len(documents)): used += [0] 
   last += len(documents)
   if (len(docs[state+1])>1):
     for n in docs[state+1][1:]:
        collection.add(
	  documents=n[1:],
	  ids=[f"id{last+i-1}" for i in range(1,len(n))],
          metadatas=[{"state": g, "next": n[0], "used": False, "source": 'None', "page": -1 } for i in range(1,len(n)) ]
        )
        for i in range(1,len(n)): used += [0]
        last += len(n)-1

state += 2
docus = load_and_split_document(PDF_PATH) + load_and_split_document(PDF_PATH2)

pages = split_text_into_chunks(docus, CHUNK_SIZE, CHUNK_OVERLAP)
document_indices = list(range(len(pages)))
for batch in batched(document_indices, 166):
        collection.add(
            ids=[f"id{last+batch[i]}" for i in range(len(batch))],
            documents=[pages[i].page_content for i in batch],
            metadatas=[dict(dict(dict(pages[i].metadata, used=False), next='None'), state='None') for i in batch],
        )
        last += len(batch)

msg = "Pomozite mi sa EON uslugama"
system_content = "Korisnik User ispod postavlja pitanje ili zadatak, upareno sa unosom koji pruža dodatni mogući kontekst. Napišite odgovor na srpskom jeziku, koji na odgovarajući način kompletira zahtev ili daje odgovor na pitanje. EON je deo usluga preduzeća United Group, koje uključuju i kablovsku mrežu za digitalnu televiziju, pristup internetu, uređaj EON SMART BOX za TV sadržaj, i fiksnu telefoniju. Korisnik se može obratiti Call centru telefonom na 080012341234, ili otiči do najbliže filijale ako mu ne pomaže odgovor."
#system_content = "Zoveš se Assistant. Ogovaraj ljubazno na pitanja i probleme koje postavlja korisnik koji se zove User."
#msg_content = "odgovori mi na ovo pitanje ili problem kroz sumiranje sledećih delova uputstava, od kojih se samo neka odnose na postavljeno pitanje ili problem: "
msg_content = "dodatni mogući kontekts je: "
ns = cs = "s0"
messages = context = []
reply = ctx = ""
print ("Recite kako Vam mogu pomoći ...")
ctx_flag = True
previous = ""
while True:
  print()
  show = False
  msg = input("User> ")
  print()
  if msg=="###": 
     show = True
     print("ctx: ", ctx, ", len=", len(context))
     print("reply=", reply)
     print("msg: ", messages)
     msg = input("User> ")
     print()
     if msg=="###": break
  query_results = collection.query(
     query_texts = [ msg ],
     n_results = max_results,
     where = { "$and": [ { "$or": [ {"state": cs },  { "page": { "$nin": [ -1 ] } } ] } , { "used": False } ] }
  )
  ln = len(query_results["documents"])
  ctx = ""
  context = []
  for n in range(ln):
     context += query_results["documents"][n]
  for n in range(len(context)):
     ctx +=  context[n] + ", "
  ctx = ctx[:-2]
  if messages==[]:
     messages=[
      {"role": "system", "content": system_content},
      {"role": "user", "content": f"{msg} - " + msg_content + ctx + "."}
     ]
  else:
     if ctx_flag:
        new_message = [
           {"role": "user", "content": f"{msg}" + msg_content + ctx + "."}
        ]
     else:
       new_message = [
           #        {"role": "system", "content": [system_content] + context},
           #        {"role": "user", "content": f"{msg}" + msg_content + ctx + "."}
           {"role": "user", "content": msg + "."}
       ]
     messages.append(new_message)
  if (ln==0):
    collection.update(
       ids=[f"id{id}" for id in range(collection.count())],
       metadatas=[{"used": False} for i in range(collection.count())],
    )
    reply = "Kako Vam još mogu pomoći?"
    ctx_flag = True
    ns = cs = "s0"
  elif query_results["documents"][0][0]=="###":
    ctx_flag = False
  else:
    if (query_results["distances"][0][0] > 0.6):
       ctx_flag = True
       print("Malo je nejasno šta je problem..")
       collection.update(
          ids=[f"id{id}" for id in range(collection.count())],
          metadatas=[{"used": False} for i in range(collection.count())],
       )
       reply = "Kako Vam još mogu pomoći?"
       ns = cs = "s0"
    else:
       ctx_flag = False
       reply=query_results["documents"][0][0]
    id = query_results["ids"][0]
    meta = query_results["metadatas"][0]
    # ovo treba usloviti potrebnom input var ako treba
    if (meta[0]["state"] not in ['None','s99']): meta[0]["used"] = True
    collection.update( ids=id, metadatas=meta)
    ns = query_results["metadatas"][0][0]["next"]
    if show: print(cs, "->", ns)
  if not(show):
    print(f"Assistant({cs}): ")
  else:
    print(f"Assistant({cs}): ", query_results["documents"][0][0]) ; print(messages)
  completion = oc.chat.completions.create(
    model = model_id,
    messages=messages,
    temperature=0.6,        #0.9
    max_tokens=2048,
    top_p=0.8,              #0.2, 0.92
    frequency_penalty=0.0, #0.05
    presence_penalty=0.0,   # 1.11
    stop=None,
#   do_sampe=True,
#    stream=True
  )
  new_message = {"role": "assistant", "content": ""}
  new=completion.choices[0].message.content
  if new==previous:
     new_message["content"] = reply
     collection.update(
        ids=[f"id{id}" for id in range(collection.count())],
        metadatas=[{"used": False} for i in range(collection.count())],
     )
     previous = reply = "Kako Vam još mogu pomoći?"
     printf (reply, end="", flush=True)
     ns = cs = "s0"
     messages=[]
  else:
     new_message["content"] += completion.choices[0].message.content
     previous = new
     print(new, end="", flush=True)
  messages.append(new_message)
  #messages=[]
  if (ns!='None'):
    if (ns != cs): cs=ns ; context = [] ; ctx = "" ; ctx_flag = True
  if cs=='None': cs='s0' ; ctx_flag = False

print("Kraj.")

