#!/usr/bin/python
# -*- coding: utf8 -*-
from abc import ABC, abstractmethod


class LanguageModel(ABC):
    @abstractmethod
    def score(self, sentence):
        pass
