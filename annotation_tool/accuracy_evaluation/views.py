from django.shortcuts import render
from django.template import loader
from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse


from accuracy_evaluation.models import Game, Sentence, Token

# Create your views here.

def index(request):
    games = Game.objects.all()
    template = loader.get_template('games/index.html')
    context = {
        'games': games,
    }
    return HttpResponse(template.render(context, request))

def detail(request, game_id, sentence_index=0):
    game = Game.objects.get(text_id=game_id)
    previous_sentence = None
    next_sentence = None
    all_sentences = game.sentence_set.all().order_by('index')
    current_sentence = all_sentences[sentence_index]
    if sentence_index > 0:
        previous_sentence = all_sentences[sentence_index-1]
    if sentence_index < len(all_sentences) - 1:
        next_sentence = all_sentences[sentence_index+1]

    if len(request.POST):
        for key, value in request.POST.items():
            if 'TOKEN' in key:
                _, token_id = key.split('_')
                token = Token.objects.get(id=token_id)
                token.annotation = value
                token.save()

        if 'continue' in request.POST:
            return HttpResponseRedirect(reverse('games:detail', args=(game_id, sentence_index+1)))


    tokens = current_sentence.token_set.all()
    template = loader.get_template('games/detail.html')
    label_set = Token.ANNOTATION_CHOICES
    context = {
        'game': game,
        'previous_sentence': previous_sentence,
        'current_sentence': current_sentence,
        'next_sentence': next_sentence,
        'tokens': tokens,
        'label_set': label_set,
        'previous_index': sentence_index-1 if sentence_index > 0 else 0,
        'num_sentences': len(all_sentences),
        'current_sent_index': sentence_index+1
    }
    return HttpResponse(template.render(context, request))
