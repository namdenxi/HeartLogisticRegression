# from django.shortcuts import render
from audioop import reverse
from django import forms
from django.shortcuts import redirect, render
from django.contrib.auth.forms import UserCreationForm, UsernameField
from django.contrib.auth.views import LoginView
from django.contrib.auth.models import User
from django.views.generic import TemplateView, FormView
# Create your views here.
class HomeView(TemplateView):
    template_name = "index.html"

class Login(LoginView):
    template_name = "login.html"

class RegisterForm(UserCreationForm):
    class Meta:
        model = User
        fields = ('username', 'email')
        fields_classes = {'username': UsernameField}
        widgets = {
            'email': forms.EmailInput(attrs={'required': True})
        }

class Register(FormView):
    template_name = "register.html"
    form_class = RegisterForm
    
    def form_valid(self, form):
        data = form.cleaned_data
        User.objects.create_user(
            username= data['username'],
            email= data['email'],
            password= data['password1']
        )
        return redirect('Login')

def HomeIndex(request):
    return render(request, 'index.html')

def Chat(request):
    return render(request, 'Chat.html')

def error(request):
    return render(request, 'error.html')
