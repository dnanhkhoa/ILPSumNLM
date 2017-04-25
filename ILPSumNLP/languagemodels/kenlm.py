#!/usr/bin/python
# -*- coding: utf8 -*-
import kenlm
from languagemodels import *


class Kenlm(LanguageModel):
    def __init__(self, model_file):
        self.model = kenlm.Model(model_file)

    def score(self, sentence):
        super().score(sentence)
