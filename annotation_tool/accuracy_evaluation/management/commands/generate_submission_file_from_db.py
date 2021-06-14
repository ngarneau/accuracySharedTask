import joblib
import locale
import json
import os
from django.core.management.base import BaseCommand, CommandError
from datetime import datetime as dt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from nltk.corpus import stopwords

from accuracy_evaluation.models import Game, Sentence, Token


class Command(BaseCommand):
    help = 'Generate submission file from database annotations'

    def handle(self, *args, **options):
        games = Game.objects.all()
        errors = list()
        for game in games:
            for y, sentence in enumerate(game.sentence_set.all().order_by('index')):
                token_start = 0
                token_end = 0
                in_error = False
                error_name = "NONE"
                error_tokens = list()
                for i, token_obj in enumerate(sentence.token_set.all().order_by('index')):
                    token = token_obj.text
                    error = token_obj.annotation
                    if error != "NONE":
                        if not in_error:
                            token_start = i+1
                        error_name = error
                        error_tokens.append(token)
                        in_error = True
                    else:
                        if in_error:  # We jumped out of an error, insert error in list and reset pointers
                            token_end = i
                            errors.append((
                                game.text_id + '.txt',
                                y+1,
                                None,
                                " ".join(error_tokens),
                                token_start,
                                token_end,
                                None,
                                None,
                                error_name,
                                None,
                                None
                            ))
                            error_name = "NONE"
                            in_error = False
                            error_tokens = list()

        df = pd.DataFrame(errors, columns=["TEXT_ID", "SENTENCE_ID", "ANNOTATION_ID", "TOKENS", "SENT_TOKEN_START", "SENT_TOKEN_END", "DOC_TOKEN_START", "DOC_TOKEN_END", "TYPE", "CORRECTION", "COMMENT"])
        df.to_csv('../data/submissions/database/submission.csv', index=False)

        self.stdout.write(self.style.SUCCESS(f'Successfully ran command "{self.help}"'))
