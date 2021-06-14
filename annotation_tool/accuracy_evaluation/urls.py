from django.urls import path

from . import views

app_name = 'games'

urlpatterns = [
    path('', views.index, name='index'),
    path('<str:game_id>/<int:sentence_index>', views.detail, name='detail'),
]
