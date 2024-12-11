from rest_framework import serializers
from django.contrib.auth import get_user_model
from .models import MedicalData
from rest_framework import serializers
from .models import Doctor

User = get_user_model()
class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = [
            'id', 'name', 'email', 'password', 
            'age', 'gender', 'diagnosis', 'treatment', 'medication'
        ]
        extra_kwargs = {
            'password': {'write_only': True},
            'email': {'required': True},
            'id': {'read_only': True},
            'name': {'required': False},
            'age': {'required': False, 'allow_null': True},
            'gender': {'required': False, 'allow_blank': True},
            'diagnosis': {'required': False, 'allow_blank': True},
            'treatment': {'required': False, 'allow_blank': True},
            'medication': {'required': False, 'allow_blank': True},
        }

    def validate(self, data):
        if self.context.get('action') == 'signup' and not data.get('name'):
            raise serializers.ValidationError({"name": "This field is required during signup."})
        return data

    def create(self, validated_data):
        validated_data["username"] = "222"
        user = User.objects.create_user(**validated_data)
        return user

    def update(self, instance, validated_data):
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance

class MedicalDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalData
        fields = [
            'user', 'test_results', 'A1', 'A2', 'A3', 'A4', 'A5', 
            'A6', 'A7', 'A8', 'A9', 'A10', 'additional_info', 
            'created_at', 'updated_at'
        ]
        extra_kwargs = {
            'user': {'read_only': True},  
            'test_results': {'required': False}, 
            'additional_info': {'required': False},  
            'created_at': {'read_only': True},  
            'updated_at': {'read_only': True},  
        }

    def create(self, validated_data):
        user = self.context['request'].user
        validated_data['user'] = user
        return super().create(validated_data)

    def update(self, instance, validated_data):
        for attr, value in validated_data.items():
            setattr(instance, attr, value)
        instance.save()
        return instance
    


class DoctorSerializer(serializers.ModelSerializer):
    class Meta:
        model = Doctor
        fields = '__all__'

from .models import MedicalData

class MedicalDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MedicalData
        fields = ['user', 'test_results', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'additional_info', 'ai_detection']
        extra_kwargs = {
            'test_results': {'required': False},  
            'additional_info': {'required': False},  
            'ai_detection': {'read_only': True, 'required':True},  
      }
