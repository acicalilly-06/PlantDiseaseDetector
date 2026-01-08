from django.urls import path
from . import views

urlpatterns = [
    # 🧠 Endpoint for uploading image and predicting
    path('predict/', views.api_predict, name='api_predict'),

]
