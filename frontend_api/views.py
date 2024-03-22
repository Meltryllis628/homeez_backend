from django.shortcuts import render
from rest_framework.views import APIView
from rest_framework.response import Response
from django.http import FileResponse
from rest_framework import status
import json

import datetime

from .serializer import FurnishingRequestSerializer, FurnishingRequestGetSerializer, FurnishingRequestJsonSerializer
from .models import FurnishingRequest

# Create your views here.
class FurnishingRequestView(APIView):
    def get(self, request, format=None):
        serializer = FurnishingRequestSerializer()
        requests = FurnishingRequest.objects.all()
        serializer = FurnishingRequestGetSerializer(requests, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)
 
    def post(self, request, format=None):
        serializer = FurnishingRequestSerializer(data=request.data)
        if serializer.is_valid():
            # user = request.user
            request_time = datetime.datetime.now()
            expire_time = request_time + datetime.timedelta(days=7)
            # serializer.validated_data['user'] = user
            serializer.validated_data['expire_time'] = expire_time
            saved_obj =  serializer.save()
            response_json = {
                "request_id": saved_obj.request_id,
                "expire_time": expire_time
            }
            return Response(response_json, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
class FurnishingRequestJsonGetView(APIView):
    def post(self, request, format=None):
        serializer = FurnishingRequestJsonSerializer(data=request.data)
        if serializer.is_valid():

            saved_obj = serializer.save()
            response_json = {
                "request_id": saved_obj.request_id,
                "expire_time": expire_time
            }
            return Response(response_json, status=status.HTTP_201_CREATED)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class FurnishingRequestDetailView(APIView):
    def get(self, request, request_id, format=None):
        try:
            request = FurnishingRequest.objects.get(request_id=request_id)
            serializer = FurnishingRequestGetSerializer(request)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except FurnishingRequest.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
    def delete(self, request, request_id, format=None):
        try:
            request = FurnishingRequest.objects.get(request_id=request_id)
            request.delete()
            return Response(status=status.HTTP_204_NO_CONTENT)
        except FurnishingRequest.DoesNotExist:
            return Response(status=status.HTTP_404_NOT_FOUND)
        
class FurnishingRequestDownloadImageView(APIView):
    def get(self, request, request_id, format=None):
        try:
            request = FurnishingRequest.objects.get(request_id=request_id)
            file_path_image = request.output_file_image.path
            file_response_image = FileResponse(open(file_path_image, 'rb'))
            return file_response_image
        except:
            return Response("Request not found or not accomplished.",status=status.HTTP_404_NOT_FOUND)

class FurnishingRequestDownloadJsonView(APIView):
    def get(self, request, request_id, format=None):
        try:
            request = FurnishingRequest.objects.get(request_id=request_id)
            file_path_json = request.output_file_json.path
            file_response_json = FileResponse(open(file_path_json, 'rb'))
            return file_response_json
        except:
            return Response("Request not found or not accomplished.",status=status.HTTP_404_NOT_FOUND)

class FurnishingRequestDownloadInputJsonView(APIView):
    def get(self, request, request_id, format=None):
        try:
            request = FurnishingRequest.objects.get(request_id=request_id)
            file_path_json = request.input_file_json.path
            file_response_json = FileResponse(open(file_path_json, 'rb'))
            return file_response_json
        except:
            return Response("Request not found or not accomplished.",status=status.HTTP_404_NOT_FOUND)

class FurnishingRequestDownloadRoomJsonView(APIView):
    pass