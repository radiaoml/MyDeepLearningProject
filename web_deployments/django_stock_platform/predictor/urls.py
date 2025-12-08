from django.urls import path
from . import views

urlpatterns = [
    path('', views.render_home_page, name='index'),
    path('predict/', views.process_stock_prediction, name='predict'),
]
