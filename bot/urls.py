from django.urls import path
from . import views
#from . import tests

urlpatterns = [
    path('', views.index, name='index'),
    path('bot/', views.editor, name='bot'),
]

urlpatterns += [
    path('reset/', views.reset, name='reset'),
]

urlpatterns += [
    path('languages/', views.BotView.as_view(), name='languages'),
    path('language/<int:pk>', views.BotViewDetail.as_view(), name='bot-detail'),
    path('bot/<int:pk>/<int:pk2>', views.editor_id, name='editor_id'),
    path('params/', views.params, name='params'),
    path('params/0', views.params_refresh, name='params-refresh'),
    path('importpdf', views.importPdf, name='importpdf'),
#    path('bot/create', views.editor_create, name='editor_create'),
#    path('bot/delete', views.editor_delete, name='editor_delete'),
]

#urlpatterns += [
#    path('tests/', views.LanguageViewTest.as_view(), name='tests'),
#]

#urlpatterns += [
#    path('tests/', views.UserViewTest.as_view(), name='tests'),
#]
