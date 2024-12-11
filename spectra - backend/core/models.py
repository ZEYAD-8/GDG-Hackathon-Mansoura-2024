from django.db import models
from django.contrib.auth.models import User
# Create your models here.

from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    name = models.CharField(max_length=255)
    age = models.PositiveIntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female')])
    diagnosis = models.TextField(null=True, blank=True)
    treatment = models.TextField(null=True, blank=True)
    medication = models.TextField(null=True, blank=True)
    email = models.EmailField(unique=True)
    password = models.CharField()
    username=models.CharField(null=True, blank=True, default="111")
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name', 'password']


    groups = models.ManyToManyField(
        'auth.Group',
        related_name='core_user_set',
        blank=True,
        help_text='The groups this user belongs to.',
        related_query_name='user',
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission',
        related_name='core_user_permissions_set',
        blank=True,
        help_text='Specific permissions for this user.',
        related_query_name='user',
    )

class MedicalData(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='medical_data')
    test_results = models.JSONField(null=True, blank=True)
    A1 = models.BooleanField(default=False)
    A2 = models.BooleanField(default=False)
    A3 = models.BooleanField(default=False)
    A4 = models.BooleanField(default=False)
    A5 = models.BooleanField(default=False)
    A6 = models.BooleanField(default=False)
    A7 = models.BooleanField(default=False)
    A8 = models.BooleanField(default=False)
    A9 = models.BooleanField(default=False)
    A10 = models.BooleanField(default=False)
    ai_detection = models.BooleanField(default=False)
    severity = models.CharField(max_length=20, choices=[('mild', 'Mild'), ('moderate', 'Moderate'), ('severe', 'Severe')], null=True, blank=True)
    additional_info = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)



class Roadmap(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='roadmap')
    level = models.CharField(max_length=20, choices=[('mild', 'Mild'), ('moderate', 'Moderate'), ('severe', 'Severe')])
    steps = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

class Doctor(models.Model):
    name = models.CharField(max_length=255)
    phone = models.CharField(max_length=20)
    location = models.CharField(max_length=255, null=True, blank=True)
    stars = models.FloatField()
    specialization = models.CharField(max_length=100, null=True, blank=True)

    hospital = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
