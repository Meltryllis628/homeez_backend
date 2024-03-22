from rest_framework import serializers
from .models import FurnishingRequest

class FurnishingRequestSerializer(serializers.ModelSerializer):

    class Meta:
        model = FurnishingRequest
        fields =  ['input_file_image', 'input_file_json']

class FurnishingRequestJsonSerializer(serializers.ModelSerializer):
    input_file_json = serializers.JSONField()
    class Meta:
        model = FurnishingRequest
        fields =  ['input_file_json']

class FurnishingRequestGetSerializer(serializers.ModelSerializer):
    class Meta:
        model = FurnishingRequest
        fields =  "__all__"