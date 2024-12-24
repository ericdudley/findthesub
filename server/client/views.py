from django.shortcuts import render
from django.http import HttpResponse, JsonResponse
from django.template import loader

def index(request):
    template = loader.get_template("client/index.html")
    return HttpResponse(template.render({
        "title": "Find The Sub"
    }))