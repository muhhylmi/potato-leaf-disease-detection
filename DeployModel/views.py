from django.http import HttpResponse
from django.shortcuts import render

def home(request):
    return render(request, "home.html")

def result(request):
	result = request.GET['file']
	return render(request, "result.html")