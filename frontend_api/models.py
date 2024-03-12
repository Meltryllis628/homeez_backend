from django.db import models
from django.contrib.auth.models import User
import uuid

# Create your models here.
class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)


class FurnishingRequest(models.Model):
    request_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    date_time = models.DateTimeField(auto_now_add=True)
    expire_time = models.DateTimeField()
    user = models.ForeignKey(User, on_delete=models.CASCADE)

    Input_file_image = models.FileField(upload_to='uploads/')
