"""
URL configuration for SmartHealth project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path
from django.views.generic import TemplateView
from home import views as homeViews
from doctor import views as doctorViews
from Diagnostic import views as heartViews
from django.contrib.auth import views as auth_views
from django.conf.urls.static import static
from django.conf import settings
from django.conf.urls import handler404, handler500


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', homeViews.HomeIndex, name="Home"),
    path('Chat', homeViews.Chat, name="Chat"),
    path('Heart', heartViews.Heart, name="Heart"),
    path('Doctor', doctorViews.Doctor, name="Doctor"),
    # path('', TemplateView.as_view(template_name="index.html")),
    # path('Chat', TemplateView.as_view(template_name="Chat.html"), name='Chat'),
    path('Login', 
         homeViews.Login.as_view(redirect_authenticated_user=True), name='Login'),
    path('Register', homeViews.Register.as_view(), name='Register'),
    path('Logout/',auth_views.LogoutView.as_view(), name='Logout')
] + static(settings.MEDIA_URL, document_root = settings.MEDIA_ROOT)

admin.site.site_header = "Smart Health Prediction"
admin.site.index_title = "Smart Health Prediction - Admin"
# handler404 = homeViews.error
# handler500 = homeViews.error