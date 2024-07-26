import bot.uchat
from django.shortcuts import render
from .models import Language, Query, ParamModelForm, Param
from bot.forms import RenewBotForm, EditorForm, EditorListForm, ParamForm, ImportForm
from django.shortcuts import get_object_or_404
from django.http import HttpResponseRedirect
from django.urls import reverse
import datetime
from django.db import transaction
from openai import OpenAI
from django.contrib.auth.decorators import login_required, permission_required
import json

#bot.uchat.chroma_client.allow_reset = True

class chat(object): 
 ns = cs = "s0"
 reply = ctx = ""
 messages = context = []
 ctx_flag = True
 previous = ""
 oc = bot.uchat.oc
 def get_reply(self,jezik,query,max_results,flag):
  msg = query # + "Objasnite na " + jezik + ", "
  query_results = bot.uchat.collection.query(
     query_texts = [ msg ],
     n_results = max_results,
     where = { "$and": [ {"$and": [ { "$or": [ {"state": self.cs },  { "page": { "$nin": [ -1 ] } } ] } , { "used": False } ] } ,
               {"lang": jezik } ] },
  )
  if bot.uchat.DEBUG: print("Query results:\n", query_results, "jezik:", jezik)
  ln = len(query_results["documents"][0])
  if ln == 0:
      if bot.uchat.DEBUG: print("Odaberite drugi jezik")
      bot.uchat.collection.update(
         ids=[f"id{id}" for id in range(bot.uchat.collection.count())],
         metadatas=[{"used": False} for i in range(bot.uchat.collection.count())],
      )
      if len(self.messages)>0: self.messages = self.messages[0]
      new = reply = "Kako Vam još mogu pomoći?"
      self.ctx_flag = True
      self.ns = self.cs = "s0"
      return new
  else:
    context = []
    i_state = l_state = -1
    for n in range(ln):
       if query_results["distances"][0][n] < (0.05+(bot.uchat.max_distance+bot.uchat.min_distance)/2):
          context += [ query_results["documents"][0][n] ]
          if i_state<0:
             l_state += 1
             if query_results["metadatas"][0][n]['state'] != 'None':
                i_state = n
       else:
          if (query_results["metadatas"][0][n]['state'] != 'None') and (i_state<0):
             i_state = n
    if l_state<0:
       l_state = i_state = 0
       ctx = query_results["documents"][0][0] 
       if self.cs != 'None':
          for n in range(1,len(context)): ctx +=  ", " + context[n]
    else:
       ctx = context[l_state]
       for n in range(len(context)):
          if l_state != n:
             ctx +=  ", " + context[n]
    ctx += "."
  match jezik:
     case 'hrvatski':
        o_jezik = 'hrvatskom'
     case 'slovenački':
        o_jezik = 'slovenačkom'
     case 'srpski':
        o_jezik = 'srpskom'
     case 'makedonski':
        o_jezik = 'makedonskom'
  if query_results["documents"][0][i_state]=="###":
    self.ctx_flag = False
    new = reply = "Kako Vam još mogu pomoći?"
    if bot.uchat.DEBUG: print("!!!###!!!")
    return "Ne razumem, ili je u pitanju greška."
  else:
    if ((query_results["distances"][0][i_state] > bot.uchat.max_distance)) and (self.cs == 'None'):
       self.ctx_flag = False
       if bot.uchat.DEBUG: print("Malo je nejasno šta je problem..")
       bot.uchat.collection.update(
          ids=[f"id{id}" for id in range(bot.uchat.collection.count())],
          metadatas=[{"used": False} for i in range(bot.uchat.collection.count())],
       )
       if len(messages)>0: messsages = messages[0]
       reply = "Malo je nejasno šta je problem - kako Vam još mogu pomoći?"
       self.ns = self.cs = "s0"
       return reply
    elif (query_results["distances"][0][i_state] < bot.uchat.min_distance):
       if self.cs != "None":
          self.ctx_flag = False
       else:
          self.ctx_flag = True
       new = reply=query_results["documents"][0][i_state]
    else:
       self.ctx_flag = True
       # reply=query_results["documents"][0][i_state]
    if (len(msg) < bot.uchat.min_len/4) or flag: self.ctx_flag = False
    id = query_results["ids"][0]
    meta = query_results["metadatas"][0]
    # ovo treba usloviti potrebnom input var ako treba
    if (meta[i_state]["state"] not in ['','s99']): meta[0]["used"] = True
    bot.uchat.collection.update( ids=id, metadatas=meta)
    self.ns = query_results["metadatas"][0][i_state]["next"]
    if self.messages==[]:
       self.messages = [ {"role": "system", "content": bot.uchat.system[jezik]} ]
       new_message = {"role": "user", "content": f"{msg} " + bot.uchat.msg_content[jezik] + " " + ctx + ". "}
    else:
       if self.ctx_flag:
          new_message = {"role": "user", "content": f"{msg}. " + bot.uchat.msg_content[jezik] + " " + ctx + ". "}
       else:
          new_message = {"role": "user", "content": f"{msg}."}
    self.messages.append(new_message)
    print("messages=", self.messages)
    if query_results["distances"][0][i_state] > bot.uchat.min_distance:
       completion = bot.uchat.oc.chat.completions.create(
          model = bot.uchat.model_id,
          messages=self.messages,
          temperature=bot.uchat.temperature,
          max_tokens=bot.uchat.max_tokens,
          top_p=bot.uchat.top_p,
          frequency_penalty=bot.uchat.frequency_penalty,
          presence_penalty=bot.uchat.presence_penalty,
          stop=["###","<|","|im_end|","|im_start|"],
          #   do_sample=True,
       )
       if bot.uchat.DEBUG: print('msg=' + new_message['content'] + "\nMessages:\n", self.messages, flush=True)
       new_message = {"role": "assistant", "content": ""}
       new=completion.choices[0].message.content
       if new==self.previous:
          if bot.uchat.DEBUG: print("Same!!!")
          bot.uchat.collection.update(
             ids=[f"id{id}" for id in range(bot.uchat.collection.count())],
             metadatas=[{"used": False} for i in range(bot.uchat.collection.count())],
          )
          new_message["content"] = new = self.previous = reply = "Kako Vam još mogu pomoći?"
          self.ns = self.cs = "s0"
          self.messages = self.messages[0]
       else:
          new_message["content"] += completion.choices[0].message.content
          self.previous = new_message["content"]
          if bot.uchat.DEBUG: print(new, end="", flush=True)
    else:
       new_message = {"role": "assistant", "content": ""}
       new_message["content"] = reply
       new = reply
    self.messages.append(new_message)
    # Prevođenje na druge jezike:
    #match o_jezik:
    #   case 'srpskom':
    #       pass
    #   case _ :
    #       new_message = {"role": "user", "content": f"Napiši to na {o_jezik} jeziku."}
    #       self.messages.append(new_message)
    #       completion = bot.uchat.oc.chat.completions.create(
    #          model = bot.uchat.model_id,
    #          messages=self.messages,
    #          temperature=bot.uchat.temperature,
    #          max_tokens=bot.uchat.max_tokens,
    #          top_p=bot.uchat.top_p,
    #          frequency_penalty=bot.uchat.frequency_penalty,
    #          presence_penalty=bot.uchat.presence_penalty,
    #          stop=None,
    #          #   do_sample=True,
    #       )
    #       new=completion.choices[0].message.content
    #       new_message = {"role": "user", "content": new}
    #       self.messages.append(new_message)
    if (self.ns!='None'):
      if (self.ns != self.cs): self.cs=self.ns 
  if self.cs=='None': self.cs='s0' ; self.ctx_flag = False
  return new

def selected(jez,str):
    if jez==str:
       return 'selected'
    else:
       return ' '

def index(request):
    #num_books = Book.objects.all().count()
    # Number of visits to this view, as counted in the session variable.
    chat_inst = chat()
    request.session.set_expiry(300)
    jezik = request.session.get('jezik', 'srpski')
    num_visits = request.session.get('num_visits', 1)
    if num_visits == 0:
        if bot.uchat.DEBUG: print("Debug: ")
    request.session['num_visits'] = num_visits+1
    reply = request.session.get("replies", "Zdravo, kako Vam mogu pomoći?")
    if request.method == 'POST':
        # Create a form instance and populate it with data from the request (binding):
        form = RenewBotForm(request.POST)
        if form.is_valid():
           request.session['query'] = query = form.cleaned_data['query']
           jezik = form.cleaned_data['jezik']
           request.session['jezik'] = jezik
           nr =  bot.uchat.max_results if num_visits < bot.uchat.max_results else 1+(num_visits % 2)
           #iif len(chat_inst.messages)>0:
           #    request.session['messages'] = chat_inst.messages if num_visits % 2 == 1 else chat_inst.messages[0:0]
           chat_inst.messages = request.session.get('messages',[])
           chat_inst.cs = request.session.get('cs','s0')
           chat_inst.ns = request.session.get('ns','s0')
           if bot.uchat.max_conv < len(chat_inst.messages):
              new_reply = chat_inst.get_reply(jezik, query, nr, True)
           else:
              new_reply = chat_inst.get_reply(jezik, query, nr, False)
           request.session['messages'] = chat_inst.messages
           request.session['ns'] = chat_inst.ns
           request.session['cs'] = chat_inst.cs
           request.session['replies'] = reply + "\nKorisnik: " + query + "\n\nOdgovor: " + new_reply + "\n"
#           q = QueryForm(request.POST)
           return HttpResponseRedirect(reverse("index"))
        else:
           if bot.uchat.DEBUG: print("Invalid form!", form.errors)
    else:
        form = RenewBotForm(initial={'query': 'Neki upit:'})
        query = request.session.get('query','...')
        jezik = request.session.get('jezik','srpski')
        #if len(chat_inst.messages)>0:
        #    request.session['messages'] = chat_inat.messages
        #else:
        #    request.session['messages'] = []
        return render(
        request,
        'index.html',
        context={
                 'num_visits': num_visits,
                 'reply': reply,
                 'selected_hr': selected(jezik,'hrvatski'),
                 'selected_sl': selected(jezik,'slovenački'),
                 'selected_sr': selected(jezik,'srpski'),
                 'selected_mk': selected(jezik,'makedonski'),
                }
    )

def reset(request):
    chat().messages = request.session.get('messages',[])
    chat().cs = "s0"
    chat().ns = "s0"
    chat().oc.close()
    bot.uchat.oc = oc = OpenAI(base_url="http://localhost:4891/v1", api_key="not-needed")
    ids = bot.uchat.collection.get()['ids']
    for id in ids: bot.uchat.collection.update(ids=[id], metadatas=[{'used': False}])
    jezik = request.session.get('jezik','srpski')
    request.session.flush()
    request.session['num_visits'] = 0
    return render(
        request,
        'index.html',
        context = {'num_visits': 0,
                   'reply': 'Kako Vam mogu još pomoći?',
                 'selected_hr': selected(jezik,'hrvatski'),
                 'selected_sl': selected(jezik,'slovenački'),
                 'selected_sr': selected(jezik,'srpski'),
                 'selected_mk': selected(jezik,'makedonski'),
                }
        )

def editor(request):
    #pagesize = 10
    if request.method == "POST":
        form = EditorListForm(request.POST)
        form.is_valid()
        try:
           request.session['id'] = f_id = form.cleaned_data['f_id']
        except:
           request.session['id'] = f_id = ""
        try:
           request.session['state'] = f_state = form.cleaned_data['f_state']
        except:
           request.session['state'] = f_state = ""
        try:
           request.session['next'] = f_next = form.cleaned_data['f_next']
        except:
           request.session['next'] = f_next = ""
        try:
           request.session['page'] = f_page = form.cleaned_data['f_page']
        except:
           request.session['page'] = f_page = ""
        try:
           request.session['jezik'] = f_jezik = form.cleaned_data['f_jezik']
        except:
           request.session['jezik'] = f_jezik = ""
        try:
           request.session['docu'] = f_docu = form.cleaned_data['f_docu']
        except:
           request.session['docu'] = f_docu = ""
        return HttpResponseRedirect(reverse('bot'))
    else:
       f_jezik = jezik = request.session.get('jezik','')
       f_state = request.session.get('state','')
       f_next = request.session.get('next','')
       f_page = request.session.get('page','')
       f_docu = request.session.get('docu','')
       f_id = request.session.get('id','')
       meta = {}
       if f_state != "":
              meta = {"state": f_state}
       if f_next != "":
          if meta == {}:
              meta = {"next": f_next}
          else:
              meta = { "$and": [ meta, {"next": f_next} ] }
       if f_page !="":
          if meta == {}:
              meta = {"page": f_page}
          else:
              meta = { "$and": [ meta , {"page": f_page} ] }
       if f_jezik != "":
           if meta == {}:
              meta = {"lang": f_jezik}
           else:
              meta = { "$and": [ meta, {"lang": f_jezik} ] }
       if f_id == "":
           if f_docu =="":
               ids = bot.uchat.collection.get(where=meta)['ids']
               documents = bot.uchat.collection.get(where=meta)['documents']
               states = bot.uchat.collection.get(where=meta)['metadatas']
           else:
               ids = bot.uchat.collection.get(where=meta, where_document={"$contains": f_docu })['ids']
               documents = bot.uchat.collection.get(where=meta, where_document={"$contains": f_docu })['documents']
               states = bot.uchat.collection.get(where=meta, where_document={"$contains": f_docu })['metadatas']
       else:
           if f_docu == "":
               ids = bot.uchat.collection.get(ids=[f'id{f_id}'], where=meta)['ids']
               documents = bot.uchat.collection.get(ids=[f'id{f_id}'], where=meta)['documents']
               states = bot.uchat.collection.get(ids=[f'id{f_id}'], where=meta)['metadatas']
           else:
               ids = bot.uchat.collection.get(ids=[f'id{f_id}'], where=meta, where_document={"$contains": f_docu })['ids']
               documents = bot.uchat.collection.get(ids=[f'id{f_id}'], where=meta, where_document={"$contains": f_docu })['documents']
               states = bot.uchat.collection.get(ids=[f'id{f_id}'], where=meta, where_document={"$contains": f_docu })['metadatas']
       cnt = 0
       nr = len(documents)
       id_docs = []
       for doc in documents:
            try:
                lang = states[cnt]['lang']
            except:
                lang = "srpski"
            id_docs += [ [ids[cnt], states[cnt]['state'], states[cnt]['next'], states[cnt]['page'], lang, doc] ]
            cnt += 1
       id_docs.sort(key=lambda k: eval(k[0][2:]))
       return render(
           request,
           'index_editor.html',
           context = {'last': bot.uchat.last, 'nr': nr,
                 'id_docs': id_docs,
                 'documents': documents,
                 'id': "",
                 'selected_hr': selected(jezik,'hrvatski'),
                 'selected_sl': selected(jezik,'slovenački'),
                 'selected_sr': selected(jezik,'srpski'),
                 'selected_mk': selected(jezik,'makedonski'),
                 'f_id': f_id,
                 'f_state': f_state,
                 'f_next': f_next,
                 'f_page': f_page,
                 'f_docu': f_docu,
                 'f_jezik': f_jezik,
                }
        )

def get_id(coll):
    for i in range(coll.count()):
        id = coll.get(ids=f'id{i}')['ids']
        if id==[]:
            return i
    return coll.count()

def editor_id(request, pk, pk2):
    last = bot.uchat.last
    state = next = uris = ""
    used = False
    page = -1
    if request.method == "POST":
        form = EditorForm(request.POST)
        if form.is_valid():
            id = form.cleaned_data['id'][2:]
            state = form.cleaned_data['state']
            next = form.cleaned_data['next']
            used = json.loads(form.cleaned_data['used'].lower())
            page = form.cleaned_data['page']
            docu = form.cleaned_data['docu']
            source = form.cleaned_data['source']
            jezik = form.cleaned_data['jezik']
            #request.session['jezik'] = jezik
            meta = {'state': state, 'next': next, 'used': used, 'page': page, 'source': source, 'lang': jezik}
            try:
               if pk2==0:
                  bot.uchat.collection.update(
                      ids = f'id{pk}',
                      metadatas = meta,
                      documents = docu
                  )
                  return HttpResponseRedirect(reverse('bot')+f'{pk}/0')
               elif pk2==1:
                  if bot.uchat.last>1:
                      new = get_id(bot.uchat.collection)
                      bot.uchat.collection.add(
                          ids = f'id{new}',
                          metadatas = meta,
                          documents = docu
                      )
                      bot.uchat.last += 1
                  return HttpResponseRedirect(reverse('bot')+f'{new}/0')
               elif pk2==2:
                  bot.uchat.collection.delete(ids = f'id{pk}')
                  bot.uchat.last -= 1
                  return HttpResponseRedirect(reverse('bot'))
            except:
              if bot.uchat.DEBUG: print("Errors:", form.errors)
              return HttpResponseRedirect(reverse('bot'))
        else:
            if bot.uchat.DEBUG: print("Invalid form!", form.errors)
            return HttpResponseRedirect(reverse('bot')+f'{pk}/0')
    else:
       if last>0:
           doc = bot.uchat.collection.get(ids=[f'id{pk}'])
           state = doc['metadatas'][0]['state']
           next = doc['metadatas'][0]['next']
           used = doc['metadatas'][0]['used']
           page = doc['metadatas'][0]['page']
           docu = doc['documents'][0]
           source = doc['metadatas'][0]['source']
       try:
           jezik = doc['metadatas'][0]['lang']
       except:
           jezik = request.session.get('jezik','srpski')
       return render(
           request,
           'index_editor.html',
           context = {'last': bot.uchat.last, 'nr': "",
                 'state': state,
                 'next': next,
                 'used': used,
                 'page': page,
                 'docu': docu,
                 'source': source,
                 'id': pk,
                 'selected_hr': selected(jezik,'hrvatski'),
                 'selected_sl': selected(jezik,'slovenački'),
                 'selected_sr': selected(jezik,'srpski'),
                 'selected_mk': selected(jezik,'makedonski'),
                }
        )

def params_refresh(request):
    if request.method == "POST":
       form = ParamForm(request.POST)
       form.is_valid()
       request.session["jezik"] = form.cleaned_data['jezik']
       return HttpResponseRedirect(reverse('params'))
    else:
       return HttpResponseRedirect(reverse('params'))

def params(request):
    if request.method == "POST":
       form = ParamForm(request.POST)
       jezici = []
       edit_all = Param.objects.all()
       for edit in edit_all: jezici += [edit.jezik]
       mform = ParamModelForm(request.POST)
       try:
          form.is_valid()
          request.session["jezik"] = form.cleaned_data['jezik']
          jezik = request.session["jezik"]
          bot.uchat.ctxpre = form.cleaned_data['ctxpre']
       except:
          bot.uchat.ctxpre = ""
       try:
          bot.uchat.msg_content[jezik] = form.cleaned_data['msg_content']
       except:
          bot.uchat.msg_content[jezik] = ""
       try:
          bot.uchat.system[jezik] = form.cleaned_data['system']
          bot.uchat.min_len = form.cleaned_data['min_len']
          bot.uchat.CHUNK_SIZE = form.cleaned_data['CHUNK_SIZE']
          bot.uchat.CHUNK_OVERLAP = form.cleaned_data['CHUNK_OVERLAP']
          bot.uchat.max_results = form.cleaned_data['max_results']
          bot.uchat.EMBED_MODEL = form.cleaned_data['EMBED_MODEL']
          bot.uchat.model_id = form.cleaned_data['model_id']
          bot.uchat.max_conv = form.cleaned_data['max_conv']
          bot.uchat.min_distance = form.cleaned_data['min_distance']
          bot.uchat.max_distance = form.cleaned_data['max_distance']
          bot.uchat.temperature = form.cleaned_data['temperature']
          bot.uchat.max_tokens = form.cleaned_data['max_tokens']
          bot.uchat.top_p = form.cleaned_data['top_p']
          bot.uchat.frequency_penalty = form.cleaned_data['frequency_penalty']
          bot.uchat.presence_penalty = form.cleaned_data['presence_penalty']
          bot.uchat.DEBUG = json.loads(form.cleaned_data['DEBUG'].lower())
       except:
          if bot.uchat.DEBUG: print("Errors:", form.errors)
       if jezik in jezici:
          for edit in edit_all:
              if edit.jezik == jezik:
                 edit.system = bot.uchat.system[jezik]
                 edit.min_len = bot.uchat.min_len
                 edit.CHUNK_SIZE = bot.uchat.CHUNK_SIZE
                 edit.CHUNK_OVERLAP = bot.uchat.CHUNK_OVERLAP
                 edit.max_results = bot.uchat.max_results
                 edit.EMBED_MODEL = bot.uchat.EMBED_MODEL
                 edit.model_id = bot.uchat.model_id
                 edit.max_conv = bot.uchat.max_conv
                 edit.min_distance = bot.uchat.min_distance
                 edit.max_distance = bot.uchat.max_distance
                 edit.temperature = bot.uchat.temperature
                 edit.max_tokens = bot.uchat.max_tokens
                 edit.top_p = bot.uchat.top_p
                 edit.frequency_penalty = bot.uchat.frequency_penalty
                 edit.presence_penalty = bot.uchat.presence_penalty
                 edit.DEBUG = bot.uchat.DEBUG
                 edit.jezik = jezik
                 edit.msg_content = bot.uchat.msg_content[jezik]
                 edit.ctxpre = bot.uchat.ctxpre
                 edit.save()
       else:
          mform.save()
       return HttpResponseRedirect(reverse('params'))
    else:
       jezik = request.session.get('jezik','srpski')
       if jezik == '': jezik='srpski'
       edit_all = Param.objects.all()
       for edit in edit_all:
          if edit.jezik == jezik:
             bot.uchat.system[jezik] = edit.system
             bot.uchat.ctxpre = edit.ctxpre
             bot.uchat.msg_content[jezik] = edit.msg_content
             bot.uchat.min_len = edit.min_len
             bot.uchat.CHUNK_SIZE = edit.CHUNK_SIZE
             bot.uchat.CHUNK_OVERLAP = edit.CHUNK_OVERLAP
             bot.uchat.max_results = edit.max_results
             bot.uchat.EMBED_MODEL = edit.EMBED_MODEL
             bot.uchat.model_id = edit.model_id
             bot.uchat.min_distance = edit.min_distance
             bot.uchat.max_distance = edit.max_distance
             bot.uchat.max_conv = edit.max_conv
             bot.uchat.temperature = edit.temperature
             bot.uchat.top_p = edit.top_p
             bot.uchat.max_tokens = edit.max_tokens
             bot.uchat.presence_penalty = edit.presence_penalty
             bot.uchat.frequency_penalty = edit.frequency_penalty
             bot.uchat.DEBUG = edit.DEBUG
       return render(
          request,
          'index_params.html',
          context = {'system': bot.uchat.system[jezik],
                   'ctrpre': bot.uchat.ctxpre,
                   'msg_content': bot.uchat.msg_content[jezik],
                   'min_len': bot.uchat.min_len,
                   'CHUNK_SIZE': bot.uchat.CHUNK_SIZE,
                   'CHUNK_OVERLAP': bot.uchat.CHUNK_OVERLAP,
                   'max_results': bot.uchat.max_results,
                   'EMBED_MODEL': bot.uchat.EMBED_MODEL,   # "all-MiniLM-L6-v2",
                   'model_id': bot.uchat.model_id,         # model_id = "Yugo60-GPT-GGUF.Q4_K_M.gguf"
                   'max_conv': bot.uchat.max_conv,
                   'min_distance': bot.uchat.min_distance,
                   'max_distance': bot.uchat.max_distance,
                   'temperature': bot.uchat.temperature,
                   'max_tokens': bot.uchat.max_tokens,
                   'top_p': bot.uchat.top_p,
                   'frequency_penalty': bot.uchat.frequency_penalty,
                   'presence_penalty': bot.uchat.presence_penalty,
                   'DEBUG': bot.uchat.DEBUG,
                   'selected_hr': selected(jezik,'hrvatski'),
                   'selected_sl': selected(jezik,'slovenački'),
                   'selected_sr': selected(jezik,'srpski'),
                   'selected_mk': selected(jezik,'makedonski'),
                }
        )

def importPdf(request):
    if request.method == "POST":
       form = ImportForm(request.POST)
       if form.is_valid():
          path = form.cleaned_data['path']
          jezik = form.cleaned_data['jezik']
          request.session['jezik'] = jezik
          request.session['path'] = path
          l_last = bot.uchat.last
          try:
             bot.uchat.load_docs(path, jezik)
             l_last = bot.uchat.last - l_last
             request.session['answer'] = f"Importovano {l_last} delova PDF dokumenta."
          except:
             if bot.uchat.DEBUG: print("Došlo je do problema sa importom PDF fajla.")
             request.session['answer'] = "Došlo je do problema sa importom PDF fajla."
       return HttpResponseRedirect(reverse('importpdf'))
    else:
       jezik = request.session.get('jezik','srpski')
       path = request.session.get('path','/Users/profile/...')
       answer = request.session.get('answer','')
       return render(
          request,
          'index_import.html',
          context = {'path': path,
                   'answer': answer,
                   'selected_hr': selected(jezik,'hrvatski'),
                   'selected_sl': selected(jezik,'slovenački'),
                   'selected_sr': selected(jezik,'srpski'),
                   'selected_mk': selected(jezik,'makedonski'),
                }
       )

from django.views import generic

class BotView(generic.ListView):
    """View for a bot chat."""
    if bot.uchat.DEBUG: print ("Test!")
    model = Language 
    paginate_by = 5
    last = bot.uchat.last
    
class BotViewDetail(generic.DetailView):
    model = Language

##@login_required
##@permission_required('catalog.can_mark_returned', raise_exception=True)

