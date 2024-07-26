from django.db import models
from django.urls import reverse
from django.forms import ModelForm

class Query(models.Model):
    query = models.TextField(
           max_length=1000, help_text="Unesite upit")
    reply = models.TextField(
           max_length=1000, help_text="Unesite odgovor")
    jezik = models.CharField(
		   max_length=16, help_text="Odaberite jezik!")
    def get_absolute_url(self):
        return reverse('query', args=[str(self.id)])
    def __str__(self):
        return self.text

#class QueryForm(ModelForm):
#    class Meta:
#        model = Query
#        fields = ['query', 'reply', 'jezik']

class Language(models.Model):
    name = models.CharField(max_length=16,
                            unique=True,
                            default="srpski",
                            help_text="Unesite jezik")
    lid = models.CharField(max_length=2, unique=False, default="sr", help_text="Unesite kratak naziv jezika")
    def get_absolute_url(self):
        """Returns the url to access a particular language instance."""
        return reverse('bot-detail', args=[str(self.id)])
    def __str__(self):
        """String for representing the Model object (in Admin site etc.)"""
        return self.name

class LanguageForm(ModelForm):
    class Meta:
        model = Language
        fields = ['name', 'lid']

class Param(models.Model):
    system = models.TextField(max_length=2048, unique=False, blank=False, default="System prpmpt", help_text="Enter parameter value")
    ctxpre = models.CharField(max_length=80, unique=False, blank=True, help_text="Enter parameter value")
    msg_content = models.CharField(unique=False, blank=True, max_length=80, help_text="Enter parameter value")
    min_len = models.IntegerField(unique=False, blank=False, default=40, help_text="Enter parameter value")
    CHUNK_SIZE = models.IntegerField(unique=False, blank=False, default=800, help_text="Enter parameter value")
    CHUNK_OVERLAP = models.IntegerField(unique=False, blank=False, default=50, help_text="Enter parameter value")
    max_results = models.IntegerField(unique=False, blank=False, default=3, help_text="Enter parameter value")
    EMBED_MODEL = models.CharField(max_length=256, default="embeddings model", help_text="Enter parameter value")
    model_id = models.CharField(max_length=256, unique=False, default="LLM model", blank=False, help_text="Enter parameter value")
    max_conv = models.IntegerField(unique=False, blank=False, default=3, help_text="Enter parameter value")
    min_distance = models.FloatField(unique=False, blank=False, default=0.35, help_text="Enter parameter value")
    max_distance = models.FloatField(unique=False, blank=False, default=0.6, help_text="Enter parameter value")
    temperature = models.FloatField(unique=False, blank=False, default=0.55, help_text="Enter parameter value")
    max_tokens = models.IntegerField(unique=False, blank=False, default=3072,  help_text="Enter parameter value")
    top_p = models.FloatField(unique=False, blank=False, default=0.8, help_text="Enter parameter value")
    frequency_penalty = models.FloatField(unique=False, default=0.0, blank=False, help_text="Enter parameter value")
    presence_penalty = models.FloatField(unique=False, default=0.0, blank=False, help_text="Enter parameter value")
    DEBUG = models.BooleanField(unique=False, blank=False, default=True, help_text="Enter parameter value")
    jezik = models.CharField(max_length=16, unique=False, blank=False, default="srpski", help_text="Uneti jezik")

class ParamModelForm(ModelForm):
    class Meta:
        model = Param
        fields = [ 'system', 'ctxpre', 'msg_content', 'min_len', 'CHUNK_SIZE', 'CHUNK_OVERLAP', 'max_results', 'EMBED_MODEL', 'model_id', 'max_conv', 'min_distance', 'max_distance', 'temperature', 'max_tokens', 'top_p', 'frequency_penalty', 'presence_penalty', 'DEBUG', 'jezik']
