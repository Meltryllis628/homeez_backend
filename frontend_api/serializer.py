from rest_framework import serializers
from .models import FurnishingRequest

class FurnishingRequestSerializer(serializers.ModelSerializer):

    class Meta:
        model = FurnishingRequest
        fields =  ['input_file_image', 'input_file_json']

class FurnishingRequestJsonSerializer(serializers.ModelSerializer):
    json_field = serializers.JSONField()
    class Meta:
        model = FurnishingRequest
        fields =  ['json_field']

class FurnishingRequestGetSerializer(serializers.ModelSerializer):
    class Meta:
        model = FurnishingRequest
        fields =  "__all__"