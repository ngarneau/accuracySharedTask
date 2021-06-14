from django.db import models

# Create your models here.

class Game(models.Model):
    text_id = models.CharField(max_length=4, unique=True)
    home_name = models.CharField(max_length=50)
    vis_name = models.CharField(max_length=50)
    line_id_from_test_set = models.IntegerField()
    generated_text = models.TextField()
    detokenized_generated_text = models.TextField()
    date = models.DateField()
    bref_box_link = models.TextField()
    bref_home_link = models.TextField()
    bref_vis_link = models.TextField()
    calendar_link = models.TextField()

    def __str__(self):
        return self.text_id


class Sentence(models.Model):
    game = models.ForeignKey(Game, on_delete=models.CASCADE)
    text = models.TextField()
    index = models.IntegerField()

    def __str__(self):
        return self.text


class Token(models.Model):
    ANNOTATION_CHOICES = [
        'NONE',
        'NUMBER',
        'NAME',
        'WORD',
        'OTHER',
        'NOT_CHECKABLE',
        'CONTEXT',
    ]
    sentence = models.ForeignKey(Sentence, on_delete=models.CASCADE)
    text = models.TextField()
    index = models.IntegerField()
    annotation = models.CharField(max_length=100, default='NONE')

    def __str__(self):
        return self.text
