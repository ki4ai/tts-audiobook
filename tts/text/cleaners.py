""" from https://github.com/keithito/tacotron """

'''
Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
from unidecode import unidecode
from .numbers import normalize_numbers
from .korean_normalization import txt_preprocessing

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
  ('mrs', 'misess'),
  ('mr', 'mister'),
  ('dr', 'doctor'),
  ('st', 'saint'),
  ('co', 'company'),
  ('jr', 'junior'),
  ('maj', 'major'),
  ('gen', 'general'),
  ('drs', 'doctors'),
  ('rev', 'reverend'),
  ('lt', 'lieutenant'),
  ('hon', 'honorable'),
  ('sgt', 'sergeant'),
  ('capt', 'captain'),
  ('esq', 'esquire'),
  ('ltd', 'limited'),
  ('col', 'colonel'),
  ('ft', 'fort'),
]]

_Start_Code, _ChoSung, _JungSung = 44032, 588, 28
_ChoSung_LIST = ['ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
_JungSung_LIST = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ',
                 'ㅣ']
_JongSung_LIST = ['', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ',
                 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']
_phone_LIST = ['ㅓ', 'ㅝ', 'ㅃ', 'ㅛ', 'ㅢ', 'ㄶ', 'ㅇ', 'ㅎ', 'ㅖ', 'ㅗ', 'ㅠ', 'ㅆ', 'ㅜ', 'ㅌ', 'ㄿ', 'ㅔ', 'ㅋ', 'ㄲ', 'ㅑ', 'ㄸ','ㅙ', 'ㅞ', 'ㅅ',
              'ㅘ', 'ㄻ', 'ㅍ', 'ㄳ', 'ㄼ', 'ㄹ', 'ㅄ', 'ㅡ', 'ㅈ', 'ㅂ', 'ㅣ', 'ㅟ', 'ㄽ', 'ㅐ', 'ㅀ', 'ㅕ', 'ㅒ', 'ㄷ', 'ㅏ', 'ㅊ', 'ㄺ', 'ㄴ', 'ㄱ',
              'ㅉ', 'ㄵ', 'ㅁ', 'ㄾ', 'ㅚ']


def expand_abbreviations(text):
  for regex, replacement in _abbreviations:
    text = re.sub(regex, replacement, text)
  return text


def expand_numbers(text):
  return normalize_numbers(text)


def lowercase(text):
  return text.lower()


def collapse_whitespace(text):
  return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
  return unidecode(text)


def decompose_hangul(text):
    line_dec = ""
    line = list(text.strip())

    for keyword in line:
        if re.match('.*[ㄱ-ㅎㅏ-ㅣ가-힣]+.*', keyword) is not None:
            if keyword in _phone_LIST:
                line_dec += keyword
            else:
                char_code = ord(keyword) - _Start_Code
                char1 = int(char_code / _ChoSung)
                line_dec += _ChoSung_LIST[char1]
                char2 = int((char_code - (_ChoSung * char1)) / _JungSung)
                line_dec += _JungSung_LIST[char2]
                char3 = int((char_code - (_ChoSung * char1) - (_JungSung * char2)))
                line_dec += _JongSung_LIST[char3]
        else:
            line_dec += keyword
    return line_dec


def basic_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def korean_cleaners(text):
  text = txt_preprocessing(text)
  text = decompose_hangul(text)
  return text


def korean_english_cleaners(text, normalization_language=1):
  if normalization_language == 1: # korean
    text = txt_preprocessing(text)
    text = lowercase(text)
  else:
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
  text = decompose_hangul(text)
  return text


def basic_english_cleaners(text):
  '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text


def transliteration_cleaners(text):
  '''Pipeline for non-English text that transliterates to ASCII.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = collapse_whitespace(text)
  return text


def english_cleaners(text):
  '''Pipeline for English text, including number and abbreviation expansion.'''
  text = convert_to_ascii(text)
  text = lowercase(text)
  text = expand_numbers(text)
  text = expand_abbreviations(text)
  text = collapse_whitespace(text)
  return text