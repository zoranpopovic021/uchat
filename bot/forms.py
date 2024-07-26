from django.core.exceptions import ValidationError
#from django.utils.translation import gettext_lazy as _

from django import forms

class RenewBotForm(forms.Form):
    query = forms.CharField(
            help_text="Unesite upit: ")
    jezik = forms.CharField(
            help_text="Odaberite jezik")
#    def clean_query(self):
#        data = self.cleaned_data['query']
#        data = self.cleaned_data['jezik']
#        return data

class EditorForm(forms.Form):
    id = forms.CharField(help_text="id")
    state = forms.CharField(help_text="Početno stanje")
    next = forms.CharField(help_text="Završno stanje")
    page = forms.IntegerField(help_text="Stranica")
    used = forms.CharField(help_text="Upotrebljen (Bool)")
    docu = forms.CharField(help_text="Dokument konteksta")
    source = forms.CharField(help_text="URL dokumenta")
    jezik = forms.CharField(help_text="Odaberite jezik")
#    def clean_query(self):
#        data = self.cleaned_data['source']
#        return data

class EditorListForm(forms.Form):
    f_id = forms.IntegerField(help_text="id filter")
    f_jezik = forms.CharField(help_text="jezik filter")
    f_state = forms.CharField(help_text="state filter")
    f_next = forms.CharField(help_text="next filter")
    f_page = forms.IntegerField(help_text="page filter")
    f_docu = forms.CharField(help_text="document filter")

class ParamForm(forms.Form):
          system = forms.CharField(help_text="Enter parameter value")
          ctxpre = forms.CharField(help_text="Enter parameter value")
          msg_content = forms.CharField(help_text="Enter parameter value")
          min_len = forms.IntegerField(help_text="Enter parameter value")
          CHUNK_SIZE = forms.IntegerField(help_text="Enter parameter value")
          CHUNK_OVERLAP = forms.IntegerField(help_text="Enter parameter value")
          max_results = forms.IntegerField(help_text="Enter parameter value")
          EMBED_MODEL = forms.CharField(help_text="Enter parameter value")
          model_id = forms.CharField(help_text="Enter parameter value")
          max_results = forms.IntegerField(help_text="Enter parameter value")
          max_conv = forms.IntegerField(help_text="Enter parameter value")
          min_len = forms.IntegerField(help_text="Enter parameter value")
          min_distance = forms.FloatField(help_text="Enter parameter value")
          max_distance = forms.FloatField(help_text="Enter parameter value")
          temperature = forms.FloatField(help_text="Enter parameter value")
          max_tokens = forms.IntegerField(help_text="Enter parameter value")
          top_p = forms.FloatField(help_text="Enter parameter value")
          frequency_penalty = forms.FloatField(help_text="Enter parameter value")
          presence_penalty = forms.FloatField(help_text="Enter parameter value")
          DEBUG = forms.CharField(help_text="Enter parameter value (Bool)")
          jezik = forms.CharField(help_text="Uneti jezik")

class ImportForm(forms.Form):
    path = forms.CharField(help_text="Unesite putanju do PDF dokumenta")
    jezik = forms.CharField(help_text="Unesite jezik")
