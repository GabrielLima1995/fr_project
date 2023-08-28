from django import forms 
from app.models import FaceDB

class SearchForm(forms.Form):
    search_query = forms.CharField(label='Search Images', max_length=100)
    