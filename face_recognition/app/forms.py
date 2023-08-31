from django import forms 
from app.models import FaceDB

class SearchForm(forms.Form):
    search_query = forms.CharField(label = '', max_length=100, widget=forms.TextInput(attrs={'placeholder': 'Insira o nome aqui'}))
    