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

from accuracy_evaluation.models import Game


class Command(BaseCommand):
    help = 'Generate submission file'

    def handle(self, *args, **options):
        predictions_path = '../data/predictions/pre_annotate/'
        files = os.listdir(predictions_path)
        files = [f for f in files if f.endswith('csv')]
        errors = list()
        for filename in tqdm(files):
            with open(os.path.join(predictions_path, filename)) as fhandle:
                token_start = 0
                token_end = 0
                in_error = False
                error_name = "NONE"
                error_tokens = list()
                for i, line in enumerate(fhandle):
                    token, error = line[:-1].split('\t')
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
                                filename.replace('csv', 'txt'),
                                None,
                                None,
                                " ".join(error_tokens),
                                None,
                                None,
                                token_start,
                                token_end,
                                error_name,
                                None,
                                None
                            ))
                            error_name = "NONE"
                            in_error = False
                            error_tokens = list()

        df = pd.DataFrame(errors, columns=["TEXT_ID", "SENTENCE_ID", "ANNOTATION_ID", "TOKENS", "SENT_TOKEN_START", "SENT_TOKEN_END", "DOC_TOKEN_START", "DOC_TOKEN_END", "TYPE", "CORRECTION", "COMMENT"])
        df.to_csv('../data/submissions/algorithm/submission.csv', index=False)

        self.stdout.write(self.style.SUCCESS(f'Successfully ran command "{self.help}"'))
