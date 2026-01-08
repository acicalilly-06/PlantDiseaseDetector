from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),          # Admin dashboard
    path('api/', include('api.urls')),        # Include app routes (predict, etc.)
]

# ✅ Serve uploaded media (images, etc.) locally when DEBUG=True
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
