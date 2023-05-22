from django.shortcuts import render
from .models import doctor
# Create your views here.
def Doctor(request):
    doctorList = doctor.objects.filter()
    return render(request, 'doctor.html', { 'doctorList': doctorList })