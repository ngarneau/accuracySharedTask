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


class ErrorFinder:
    def __init__(self, game_data):
        self.CWN_clf = joblib.load('./models/text_clf.joblib')
        self.CWN_classes = ['CONTEXT', 'NOT_CHECKABLE', 'WORD']
        self.game_data = game_data
        self.word_numbers = {
            # 'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'nineth', 'tenth',
            # 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten',
            # '1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
            # 'pair'
        }
        self.file_id = game_data['shared_task_text_id']
        self.summary = open(f'../data/texts/{self.file_id}.txt').read()
        self.home_name = game_data['home_name']
        self.vis_name = game_data['vis_name']
        self.home_city = game_data['home_city']
        self.vis_city = game_data['vis_city']
        self.home_line = game_data['home_line']
        self.vis_line = game_data['vis_line']
        self.box_score = game_data['box_score']
        self.first_name = self.box_score['FIRST_NAME']
        self.second_name = self.box_score['SECOND_NAME']
        self.minutes = self.box_score['MIN']
        self.day = self.game_data['day']
        month, day, year = self.day.split('_')
        self.weekday = dt(2017, int(month), int(day)).strftime("%A")


    def retrieve_names(self, tokens):
        names = set()
        for t in tokens:
            if t[0].isupper() and t.lower() not in stopwords.words('english'):
                names.add(t)
        return names


    def find_number_team_corr(self, number, name):
        if name == self.home_name:
            return number in self.home_line.values()
        if name == self.vis_name:
            return number in self.vis_line.values()
        else:
            return False

    def find_number_player_corr(self, number, name):
        # Check correspondence for first name with every numeric values
        columns_to_check = ['MIN', 'FGM', 'REB', 'FG3A', 'AST', 'FG3M', 'OREB', 'TO', 'PF', 'PTS', 'FGA', 'STL', 'FTA', 'BLK', 'DREB', 'FTM', 'FT_PCT', 'FG_PCT', 'FG3_PCT']
        for column in columns_to_check:
            for first_name, value in zip(self.first_name.values(), self.box_score[column].values()):
                if (name, number) == (first_name, value):
                    return True

            for second_name, value in zip(self.second_name.values(), self.box_score[column].values()):
                if (name, number) == (first_name, value):
                    return True
        return False

    def find_number_name_correspondence(self, number, name):
        return self.find_number_team_corr(number, name) or self.find_number_player_corr(number, name)

    def find_number_errors(self, tokens):
        errors = list()
        names = self.retrieve_names(tokens)
        for t in tokens:
            error = "NONE"
            if t.isdigit() or t in self.word_numbers:
                # Check if this number is associated to a team's score
                # Or check if this number is associated to a player's score
                correspondence = False
                for name in names:
                    if self.find_number_name_correspondence(t, name):
                        correspondence = True

                if not correspondence:
                    error = "NUMBER"
            errors.append(error)
        return errors


    def find_name_correspondence(self, name):
        if name in self.home_name:
            return True
        if name in self.vis_name:
            return True
        if name in self.home_city:
            return True
        if name in self.vis_city:
            return True
        if name in self.first_name.values():
            return True
        if name in self.second_name.values():
            return True
        if name == self.weekday:
            return True
        if name == "NBA": # Hack
            return True


    def find_name_errors(self, tokens):
        # Handling player, center, weekdays names
        errors = list()
        names = self.retrieve_names(tokens)
        for t in tokens:
            error = "NONE"
            if t[0].isupper() and t.lower() not in stopwords.words('english'):
                if not self.find_name_correspondence(t):
                    error = "NAME"
            errors.append(error)
        return errors

    def find_cwn_errors(self, sentence: str):
        # Processes a string using the clf model
        probas = self.CWN_clf.predict_proba([sentence])
        errors = { c: list() for c in self.CWN_classes}
        tf_idf = self.CWN_clf.steps[0][1]
        reg = self.CWN_clf.steps[1][1]
        for i, prob in enumerate(probas[0]):
            for w in sentence.split():
                if prob > 0.5: # Consider classifying words then
                    idx = tf_idf.vocabulary_.get(w)
                    if idx:
                        idf = tf_idf.idf_[idx]
                        coef = reg.coef_[i][idx]
                        if idf * coef > 0:
                            errors[self.CWN_classes[i]].append(self.CWN_classes[i])
                        else:
                            errors[self.CWN_classes[i]].append("NONE")
                    else:
                        errors[self.CWN_classes[i]].append("NONE")
                else:
                    errors[self.CWN_classes[i]].append("NONE")
        return errors["WORD"], errors["CONTEXT"], errors["NOT_CHECKABLE"]  # retuning them in terms of priority w.r.t the task


    def find_errors_in_sentence(self, sentence):
        tokens = sentence.split()
        number_errors = self.find_number_errors(tokens)
        name_errors = self.find_name_errors(tokens)
        word_errors, context_errors, na_errors = self.find_cwn_errors(sentence)
        errors = list()
        for num, name, word, context, na in zip(number_errors, name_errors, word_errors, context_errors, na_errors):
            if num != "NONE":
                errors.append(num)
            elif name != "NONE":
                errors.append(name)
            elif word != "NONE":
                errors.append(word)
            elif context != "NONE":
                errors.append(context)
            elif na != "NONE":
                errors.append(na)
            else:
                errors.append("NONE")
        return errors

    def find_errors(self):
        # Returns the tokenized sentence where for each token is associated a label of
        # either NONE, NUMBER, NAME, WORD, CONTEXT, NOT_CHECKABLE, OTHER
        sentences = self.summary.split(' . ')
        tokens = self.summary.split()
        errors = list()
        for i, sentence in enumerate(sentences):
            errors += self.find_errors_in_sentence(sentence)
            if i < len(sentences) - 1:
                errors += ["NONE"]
        return tokens, errors


class Command(BaseCommand):
    help = 'Imports basketball games'

    def handle(self, *args, **options):


        with open('../data/shared_task.jsonl') as fhandle:
            for line in fhandle:
                game_data = json.loads(line)
                error_finder = ErrorFinder(game_data)
                tokens, errors = error_finder.find_errors()
                with open(f"../data/predictions/pre_annotate/{error_finder.file_id}.csv", 'w') as fhandle:
                    for t, e in zip(tokens, errors):
                        fhandle.write(f"{t}\t{e}\n")


        # dataset = pd.read_csv('../data/games.csv')
        # for id, row in tqdm(dataset.iterrows()):
        #     month, day, year = row['DATE'].split('_')
        #     date_formatted = dt(2017, int(month), int(day)).strftime("%Y-%m-%d")
        #     game = Game(
        #         text_id=row['TEXT_ID'],
        #         home_name=row['HOME_NAME'],
        #         vis_name=row['VIS_NAME'],
        #         line_id_from_test_set=row['LINE_ID_FROM_TEST_SET'],
        #         generated_text=row['GENERATED_TEXT'],
        #         detokenized_generated_text=row['DETOKENIZED_GENERATED_TEXT'],
        #         date=date_formatted,
        #         bref_box_link=row['BREF_BOX'],
        #         bref_home_link=row['BREF_HOME'],
        #         bref_vis_link=row['BREF_VIS'],
        #         calendar_link=row['CALENDAR']
        #     )
        #     game.save()
        self.stdout.write(self.style.SUCCESS(f'Successfully ran command "{self.help}"'))
