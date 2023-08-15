from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path("webcam_stream/", views.webcam_stream, name="webcam_stream"),
    path("detected_object/", views.detected_object, name="detected_object"),
]
