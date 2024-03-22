from rest_framework import serializers
from .models import FurnishingRequest
from django.core.files import File
import os
import json
import datetime
class FurnishingRequestSerializer(serializers.ModelSerializer):

    class Meta:
        model = FurnishingRequest
        fields =  ['input_file_image', 'input_file_json']

class FurnishingRequestJsonSerializer(serializers.ModelSerializer):
    json_data = serializers.JSONField()
    class Meta:
        model = FurnishingRequest
        fields =  ['json_data']
    def create(self, validated_data):
        request_time = datetime.datetime.now()
        expire_time = request_time + datetime.timedelta(days=7)
        validated_data['expire_time'] = expire_time
        json_obj = validated_data.pop('json_data')
        instance = FurnishingRequest.objects.create(**validated_data)
        file_name = instance.request_id
        if not os.path.exists(f"uploads/input/"):
            os.makedirs(f"uploads/input/")
        path = f"uploads/input/"
        file_path = os.path.join(path, f"{file_name}.json")
        with open(file_path, 'w') as f:
            json.dump(json_obj, f)
        with open(file_path, 'r') as f:
            django_file = File(f)
            instance.input_file_json.save(file_path, django_file, save=True)
        instance.save()
        return instance

class FurnishingRequestGetSerializer(serializers.ModelSerializer):
    class Meta:
        model = FurnishingRequest
        fields =  "__all__"