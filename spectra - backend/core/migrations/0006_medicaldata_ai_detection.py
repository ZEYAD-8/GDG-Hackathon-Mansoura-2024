# Generated by Django 5.1.1 on 2024-12-02 12:22

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0005_alter_medicaldata_test_results'),
    ]

    operations = [
        migrations.AddField(
            model_name='medicaldata',
            name='ai_detection',
            field=models.BooleanField(default=False),
        ),
    ]