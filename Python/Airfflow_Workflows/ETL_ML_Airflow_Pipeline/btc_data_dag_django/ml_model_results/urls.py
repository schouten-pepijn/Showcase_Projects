from django.urls import path
from .views import model_results

urlpatterns = [
    path('model_results/', model_results),
]