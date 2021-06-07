import locale
import json
import os
from django.core.management.base import BaseCommand, CommandError
from datetime import datetime as dt
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import numpy as np


class Document:
    def __init__(self, id, text):
        self.text = text
        self.id = id
        self.sentence_delim = " . "
        self.splitted_text = self.__split_text()

    def __split_text(self):
        return self.text.split(self.sentence_delim)

    def get_sentence(self, id):
        return self.splitted_text[id-1]


class Command(BaseCommand):
    help = 'Create classification dataset'

    def load_documents(self):
        texts_path = '../data/texts'
        files = os.listdir(texts_path)
        documents = list()
        for filename in files:
            text = open(os.path.join(texts_path, filename), 'r').read()
            document = Document(filename, text)
            documents.append(document)
        return documents

    def handle(self, *args, **options):
        documents = self.load_documents()
        annotations = pd.read_csv('../data/gsml.csv')
        positive_lines = list()
        negative_lines = list()
        for document in documents:
            document_annotations = annotations[annotations['TEXT_ID'] == document.id]
            document_annotations_lines = set(document_annotations['SENTENCE_ID'].tolist())
            for line_id in range(1, len(document.splitted_text)+1):
                if line_id in document_annotations_lines:
                    positive_lines.append((document.get_sentence(line_id), 1))
                else:
                    negative_lines.append((document.get_sentence(line_id), 0))

        dataset = positive_lines + negative_lines
        np.random.shuffle(dataset)

        with open('../data/binary_classification.jsonl', 'w') as fhandle:
            for line, label in dataset:
                if label == 1.0:
                    sample = {"text": line, "label": "positive"}
                else:
                    sample = {"text": line, "label": "negative"}
                fhandle.write("{}\n".format(json.dumps(sample)))

        self.stdout.write(self.style.SUCCESS(f'Successfully ran command "{self.help}"'))
