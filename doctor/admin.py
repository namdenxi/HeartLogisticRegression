from django.contrib import admin
from .models import doctor

# Register your models here.
class DoctorAdmin(admin.ModelAdmin):
    list_display = ('doctor_ID', 'doctor_Name', 'doctor_Department', 'doctor_Position', 'doctor_Email', 'doctor_Contact', 'doctor_Address', 'doctor_Avatar')
    search_fields = ['doctor_ID', 'doctor_Name', 'doctor_Department', 'doctor_Position', 'doctor_Email', 'doctor_Contact', 'doctor_Address']
    list_filter = ('doctor_ID', 'doctor_Name', 'doctor_Department', 'doctor_Position', 'doctor_Email', 'doctor_Contact', 'doctor_Address', 'doctor_Avatar')
admin.site.register(doctor, DoctorAdmin)