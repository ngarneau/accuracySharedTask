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

    def add_arguments(self, parser):
        parser.add_argument('games_data_path', type=str)

    def handle(self, *args, **options):
        games_data_path = options['games_data_path']
        dataset = pd.read_csv(games_data_path)
        for id, row in tqdm(dataset.iterrows()):
            month, day, year = row['DATE'].split('_')
            year = int('20'+year)
            date_formatted = dt(year, int(month), int(day)).strftime("%Y-%m-%d")
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
                bref_vis_link=row['BREF_VISITING'],
                calendar_link=row['CALANDER']
            )
            game.save()
        self.stdout.write(self.style.SUCCESS(f'Successfully ran command "{self.help}"'))
