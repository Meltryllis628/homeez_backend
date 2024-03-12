from django.db import models


class FurnitureType(models.Model):
    type = models.CharField(max_length=127, primary_key=True)
    area_restroom = models.BooleanField()
    area_kitchen = models.BooleanField()
    area_bedroom = models.BooleanField()
    area_livingroom = models.BooleanField()
    furnitures = models.ForeignKey('Furniture', on_delete=models.CASCADE)

# Create your models here.
class Furniture(models.Model):
    name = models.CharField(max_length=127, primary_key=True)
    material = models.CharField(max_length=127)
    type = models.ForeignKey(FurnitureType, on_delete=models.CASCADE)
    display_name = models.CharField(max_length=127)
    theme = models.CharField(max_length=127)
    description = models.TextField()
    dimensions = models.CharField(max_length=127)
    first_category = models.CharField(max_length=127)
    second_category = models.CharField(max_length=127)
    image_link = models.CharField(max_length=127)
    s3_link = models.CharField(max_length=127)
