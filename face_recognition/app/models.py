from django.db import models

# Create your models here.

class FaceDB(models.Model):
    name = models.CharField(max_length=100)
    id = models.AutoField(primary_key=True)
    record_date = models.DateField(auto_now_add=True)
    image = models.ImageField(upload_to = 'images/')

    def __str__(self):
        return str(self.record_date)

