from django.apps import AppConfig
from dotenv import load_dotenv


class ServerConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'server'

    def ready(self):
        load_dotenv()