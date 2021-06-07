import locale
import json
from django.core.management.base import BaseCommand, CommandError
from datetime import datetime as dt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm

from accuracy_evaluation.models import Game


class Command(BaseCommand):
    help = 'Imports basketball games'

    def handle(self, *args, **options):
        dataset = pd.read_csv('../data/games.csv')
        for id, row in tqdm(dataset.iterrows()):
            month, day, year = row['DATE'].split('_')
            date_formatted = dt(2017, int(month), int(day)).strftime("%Y-%m-%d")
            game = Game(
                text_id=row['TEXT_ID'],
                home_name=row['HOME_NAME'],
                vis_name=row['VIS_NAME'],
                line_id_from_test_set=row['LINE_ID_FROM_TEST_SET'],
                generated_text=row['GENERATED_TEXT'],
                detokenized_generated_text=row['DETOKENIZED_GENERATED_TEXT'],
                date=date_formatted,
                bref_box_link=row['BREF_BOX'],
                bref_home_link=row['BREF_HOME'],
                bref_vis_link=row['BREF_VIS'],
                calendar_link=row['CALENDAR']
            )
            game.save()
        self.stdout.write(self.style.SUCCESS(f'Successfully ran command "{self.help}"'))
