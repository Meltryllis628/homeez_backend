from django.urls import re_path
from rest_framework.urlpatterns import format_suffix_patterns
from . import views

urlpatterns = [
    re_path(r'^furnishing_request/$', views.FurnishingRequestView.as_view()),
    re_path(r'^furnishing_request/(?P<request_id>[0-9a-f-]+)/$', views.FurnishingRequestDetailView.as_view()),
    re_path(r'^furnishing_request/download_img/(?P<request_id>[0-9a-f-]+)/$', views.FurnishingRequestDownloadImageView.as_view()),
    re_path(r'^furnishing_request/download_json/(?P<request_id>[0-9a-f-]+)/$', views.FurnishingRequestDownloadJsonView.as_view()),
    re_path(r'^furnishing_request/download_input_json/(?P<request_id>[0-9a-f-]+)/$', views.FurnishingRequestDownloadInputJsonView.as_view()),
    re_path(r'^furnishing_request/download_json/(?P<request_id>[0-9a-f-]+)/(?P<room>[0-9a-z-]+)', views.FurnishingRequestDownloadRoomJsonView.as_view()),
    re_path(r'^furnishing_request/json/$', views.FurnishingRequestJsonGetView.as_view())
]

urlpatterns = format_suffix_patterns(urlpatterns) 