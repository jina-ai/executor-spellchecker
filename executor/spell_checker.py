__copyright__ = "Copyright (c) 2021 Jina AI Limited. All rights reserved."
__license__ = "Apache-2.0"

import os
import pickle
from typing import Dict, Iterable

from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
from .pyngramspell import PyNgramSpell

cur_dir = os.path.dirname(os.path.abspath(__file__))


class SpellChecker(Executor):
    """A simple spell checker based on BKTree

    It can be trained on your own corpus, on the /train endpoint

    Otherwise it automatically spell corrects your Documents with string contents.
    The content is overridden.
    """

    def __init__(
        self,
        model_path: str = os.path.join(cur_dir, 'model.pickle'),
        traversal_paths: Iterable = ['r'],
        *args,
        **kwargs,
    ):
        """
        :param model_path: the path where the model will be saved
        :param traversal_paths: the path to traverse docs when processed
        """
        super().__init__(*args, **kwargs)
        self.traversal_paths = traversal_paths
        self.logger = JinaLogger(self.metas.name)

        self.model_path = model_path
        self.model = None

        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as model_file:
                try:
                    self.model = pickle.load(model_file)
                except ModuleNotFoundError as e:
                    # can happen if there is a model file built
                    # because of python importing errors
                    self.logger.warning(f'Error trying to load existing model, '
                                        f'skipping: {e}')
        else:
            self.logger.warning(f'model_path {self.model_path} is empty. Use /train')

    @requests(on='/train')
    def train(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """
        Re-train the BKTree model

        :param parameters: are passed as **kwargs to PyNgramSpell model
        """
        self.model = PyNgramSpell(**parameters)
        input_training_data = [d.content for d in docs]
        self.model.fit(input_training_data)
        self.model.save(self.model_path)

    @requests(on=['/index', '/search', '/update', '/delete'])
    def spell_check(self, docs: DocumentArray, parameters: Dict = {}, **kwargs):
        """
        Processes the text Documents
        """
        for d in docs.traverse_flat(
            parameters.get('traversal_paths', self.traversal_paths)
        ):
            if isinstance(d.content, str):
                d.content = self.model.transform(d.content)
