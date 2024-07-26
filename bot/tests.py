from django.test import TestCase

# Create your tests here.


from catalog.models import Language
from django.urls import reverse


class LanguageViewTest(TestCase):

    @classmethod
    def setUpTestData(cls):
         langs = [ 'Hrvatski', 'Slovenački', 'Srpski', 'Bosanski', 'Makedonski', 'Bugarski']
         for lang in langs:
            l = Language.objects.create(name=lang) ; l.save()

import datetime
from django.utils import timezone

from bot.models import Language

# Get user model from settings
from django.contrib.auth import get_user_model
User = get_user_model()

class UserViewTest(TestCase):

    def setUp(self):
        # Create two users
        test_user1 = User.objects.create_user(
            username='testuser1', password='testuser1')
        test_user2 = User.objects.create_user(
            username='testuser2', password='testuser2')

        test_user1.save()
        test_user2.save()

