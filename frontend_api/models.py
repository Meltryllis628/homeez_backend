from django.db import models
from django.contrib.auth.models import User
from django.dispatch import receiver
from django.db.models.signals import pre_delete, post_save
import uuid
import os
import json

def upload_input_path(instance, filename):
    ext = filename.split('.')[-1]
    file_name = instance.request_id
    if not os.path.exists(f"uploads/input/"):
        os.makedirs(f"uploads/input/")
    path = f"uploads/input/"
    return os.path.join(path, f"{file_name}.{ext}")

def upload_output_path(instance, filename):
    ext = filename.split('.')[-1]
    file_name = instance.request_id
    if not os.path.exists(f"uploads/output/"):
        os.makedirs(f"uploads/output/")
    path = f"uploads/output/"
    return os.path.join(path, f"{file_name}.{ext}")


# Create your models here.


class FurnishingRequest(models.Model):
    request_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    request_time = models.DateTimeField(auto_now_add=True)
    expire_time = models.DateTimeField()

    input_file_image = models.FileField(upload_to=upload_input_path, null=True, blank=True)
    input_file_json = models.FileField(upload_to=upload_input_path, null=True, blank=True)
    output_file_image = models.FileField(upload_to=upload_output_path, null=True, blank=True)
    output_file_json = models.FileField(upload_to=upload_output_path, null=True, blank=True)


@receiver(pre_delete, sender=FurnishingRequest)
def delete_related_files(sender, instance, **kwargs):
    # 获取对象中包含的文件字段，并删除文件
    file_fields = [field for field in instance._meta.get_fields() if isinstance(field, models.FileField)]
    for field in file_fields:
        if not getattr(instance, field.name):
            continue
        file_path = getattr(instance, field.name).path
        if os.path.isfile(file_path):
            os.remove(file_path)


@receiver(post_save, sender=FurnishingRequest)
def generate_result(sender, instance, **kwargs):
    try:
        input_json = json.load(open(instance.input_file_json.path))
        pass
    except Exception as e:
        pass
    