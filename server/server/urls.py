from django.urls import path
from server import views

urlpatterns = [
    path("", views.index, name="index"),
]