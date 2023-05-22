from django.db import models

# Create your models here.
class doctor(models.Model):
    doctor_ID = models.AutoField(primary_key=True)
    doctor_Name = models.CharField(max_length=50, null=False)
    doctor_Department = models.CharField(max_length=50, null=False, default=None)
    doctor_Position = models.CharField(max_length=50, null=False, default=None)
    doctor_Email = models.CharField(max_length=50)
    doctor_Contact = models.CharField(max_length=20)
    doctor_Address = models.TextField()
    doctor_Avatar = models.ImageField(upload_to='images')
    
    def __str__(self):
        return f"{self.doctor_ID}, {self.doctor_Name}, {self.doctor_Email}, {self.doctor_Contact}, {self.doctor_Address}, {self.doctor_Avatar}, {self.doctor_Department}"