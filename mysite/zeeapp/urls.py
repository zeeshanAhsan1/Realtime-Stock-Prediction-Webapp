from django.urls import path
from . import views

urlpatterns = [
    path("", views.index, name='index'),
    path("service", views.service, name='service'),
    path("index", views.index, name='index'),
    path("trend_prediction", views.trend_prediction, name='trend_prediction'),
    path("prediction", views.prediction, name='prediction'),
]