# Generated by Django 4.2.4 on 2023-08-15 15:40

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0003_delete_capturedimage'),
    ]

    operations = [
        migrations.CreateModel(
            name='CapturedImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='images/')),
                ('capture_date', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
