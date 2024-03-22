from rest_framework import serializers
from .models import FurnishingRequest
import os
import json
class FurnishingRequestSerializer(serializers.ModelSerializer):

    class Meta:
        model = FurnishingRequest
        fields =  ['input_file_image', 'input_file_json']

class FurnishingRequestJsonSerializer(serializers.ModelSerializer):
    input_file_json = serializers.JSONField()
    class Meta:
        model = FurnishingRequest
        fields =  ['json_data']
    def create(self, validated_data):
        json_data = validated_data.pop('json_data')
        instance = FurnishingRequest.objects.create(**validated_data)
        file_name = instance.request_id
        if not os.path.exists(f"uploads/input/"):
            os.makedirs(f"uploads/input/")
        path = f"uploads/input/"
        file_path = os.path.join(path, f"{file_name}.json")
        with open(file_path, 'w') as f:
            json.dump(json_data, f)
        instance.input_file_json = file_path
        instance.save()
        return instance

class FurnishingRequestGetSerializer(serializers.ModelSerializer):
    class Meta:
        model = FurnishingRequest
        fields =  "__all__"