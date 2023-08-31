import csv
import pandas as pd
import jsonlines
import os
import random
import emoji
import pyarabic.araby as araby
import html
import logging
import re
from typing import List
from farasa.segmenter import FarasaSegmenter
PREFIX_LIST = [
    "ال",
    "و",
    "ف",
    "ب",
    "ك",
    "ل",
    "لل",
    "\u0627\u0644",
    "\u0648",
    "\u0641",
    "\u0628",
    "\u0643",
    "\u0644",
    "\u0644\u0644",
    "س",
]
SUFFIX_LIST = [
    "ه",
    "ها",
    "ك",
    "ي",
    "هما",
    "كما",
    "نا",
    "كم",
    "هم",
    "هن",
    "كن",
    "ا",
    "ان",
    "ين",
    "ون",
    "وا",
    "ات",
    "ت",
    "ن",
    "ة",
    "\u0647",
    "\u0647\u0627",
    "\u0643",
    "\u064a",
    "\u0647\u0645\u0627",
    "\u0643\u0645\u0627",
    "\u0646\u0627",
    "\u0643\u0645",
    "\u0647\u0645",
    "\u0647\u0646",
    "\u0643\u0646",
    "\u0627",
    "\u0627\u0646",
    "\u064a\u0646",
    "\u0648\u0646",
    "\u0648\u0627",
    "\u0627\u062a",
    "\u062a",
    "\u0646",
    "\u0629",
]
_PREFIX_SYMBOLS = [x + "+" for x in PREFIX_LIST]
_SUFFIX_SYMBOLS = ["+" + x for x in SUFFIX_LIST]
_OTHER_TOKENS = ["[رابط]", "[مستخدم]", "[بريد]"]
NEVER_SPLIT_TOKENS = list(set(_PREFIX_SYMBOLS + _SUFFIX_SYMBOLS + _OTHER_TOKENS))

URL_REGEXES = [
    r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)",
    r"@(https?|ftp)://(-\.)?([^\s/?\.#-]+\.?)+(/[^\s]*)?$@iS",
    r"http[s]?://[a-zA-Z0-9_\-./~\?=%&]+",
    r"www[a-zA-Z0-9_\-?=%&/.~]+",
    r"[a-zA-Z]+\.com",
    r"(?=http)[^\s]+",
    r"(?=www)[^\s]+",
    r"://",
]
USER_MENTION_REGEX = r"@[\w\d]+"
EMAIL_REGEXES = [r"[\w-]+@([\w-]+\.)+[\w-]+", r"\S+@\S+"]
REDUNDANT_PUNCT_PATTERN = (
    r"([!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ【»؛\s+«–…‘]{2,})"
)

REGEX_TATWEEL = r"(\D)\1{2,}"
MULTIPLE_CHAR_PATTERN = re.compile(r"(\D)\1{2,}", re.DOTALL)

REJECTED_CHARS_REGEX = r"[^0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘]"
REJECTED_CHARS_REGEXV2 = r"[^0-9\u0621-\u063A\u0641-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘/]"

REGEX_URL_STEP1 = r"(?=http)[^\s]+"
REGEX_URL_STEP2 = r"(?=www)[^\s]+"
REGEX_URL = r"(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
REGEX_MENTION = r"@[\w\d]+"
REGEX_EMAIL = r"\S+@\S+"

CHARS_REGEX = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘"
CHARS_REGEXV2 = r"0-9\u0621-\u063A\u0640-\u066C\u0671-\u0674a-zA-Z\[\]!\"#\$%\'\(\)\*\+,\.:;\-<=·>?@\[\\\]\^_ـ`{\|}~—٪’،؟`୍“؛”ۚ»؛\s+«–…‘/"

WHITE_SPACED_DOUBLE_QUOTATION_REGEX = r'\"\s+([^"]+)\s+\"'
WHITE_SPACED_SINGLE_QUOTATION_REGEX = r"\'\s+([^']+)\s+\'"
WHITE_SPACED_BACK_QUOTATION_REGEX = r"\`\s+([^`]+)\s+\`"
WHITE_SPACED_EM_DASH = r"\—\s+([^—]+)\s+\—"

LEFT_SPACED_CHARS = r" ([\]!#\$%\),\.:;\?}٪’،؟”؛…»·])"
RIGHT_SPACED_CHARS = r"([\[\(\{“«‘*\~]) "
LEFT_AND_RIGHT_SPACED_CHARS = r" ([\+\-\<\=\>\@\\\^\_\|\–]) "

_HINDI_NUMS = "٠١٢٣٤٥٦٧٨٩"
_ARABIC_NUMS = "0123456789"
HINDI_TO_ARABIC_MAP = str.maketrans(_HINDI_NUMS, _ARABIC_NUMS)
def _remove_non_digit_repetition(text: str) -> str:
    """
    :param text:  the input text to remove elongation
    :return: delongated text
    """
    # loop over the number of times the regex matched the text
    # OLD
    # for index_ in range(len(re.findall(REGEX_TATWEEL, text))):
    #     elongation = re.search(REGEX_TATWEEL, text)
    #     if elongation:
    #         elongation_pattern = elongation.group()
    #         elongation_replacement = elongation_pattern[0]
    #         elongation_pattern = re.escape(elongation_pattern)
    #         text = re.sub(
    #             elongation_pattern, elongation_replacement, text, flags=re.MULTILINE
    #         )
    #     else:
    #         break

    # New
    text = MULTIPLE_CHAR_PATTERN.sub(r"\1\1", text)
    return text
def normalize_text(pp_text):
    # map some weired characters
    mapping = {"ھ": "ه", "گ": "ك", r'\s': " ", "٪": "%", "℃": "C", "·": ".", "…": ".",
               'ـــ': '-', 'ـ': '-', ",": "،", "\?": "؟", "“": '"', '”': '"'}
    # map indic to arabic digits
    digit_lst = ["٠", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩"]
    mapping.update({dig: "%s" % idx for idx, dig in enumerate(digit_lst)})

    for a, b in mapping.items():
        pp_text = re.sub(a, b, pp_text)

    return pp_text

def preprocess_v3(text: str) -> str:
    text1 = text
    remove_html_markup= True,
    replace_urls_emails_mentions = True,
    strip_tashkeel = True,
    strip_tatweel = True,
    insert_white_spaces = True,
    remove_non_digit_repetition = True,
    keep_emojis = True,
    replace_slash_with_dash = True,
    map_hindi_numbers_to_arabic = True,
    apply_farasa_segmentation = True,
    if keep_emojis:
        import emoji

        self_emoji = emoji
        emoji_regex = "".join(list(self_emoji.UNICODE_EMOJI_ENGLISH.keys()))
        # for (i,tt) in enumerate(self_emoji.get_emoji_unicode_dict("en").keys()):
        #     if i==1016:
        #         print(tt)
        # print(emoji_regex)
        sorted_chars_regexv2 = ''.join(sorted(CHARS_REGEXV2))
        sorted_emoji_regex = ''.join(sorted(emoji_regex))
        # print(sorted_chars_regexv2)
        self_REJECTED_CHARS_REGEX = "[^%s%s]" % (CHARS_REGEXV2,emoji_regex)
        print(self_REJECTED_CHARS_REGEX)
        # self_REJECTED_CHARS_REGEX = '[^%s%s]' % (sorted_chars_regexv2, sorted_emoji_regex)
        # print(self_REJECTED_CHARS_REGEX )

    text = str(text)
    text = html.unescape(text)
    if strip_tashkeel:
        text = araby.strip_tashkeel(text)
    if strip_tatweel:
        text = araby.strip_tatweel(text)

    if replace_urls_emails_mentions:
        # replace all possible URLs
        for reg in URL_REGEXES:
            text = re.sub(reg, " [رابط] ", text)
        # REplace Emails with [بريد]
        for reg in EMAIL_REGEXES:
            text = re.sub(reg, " [بريد] ", text)
        # replace mentions with [مستخدم]
        text = re.sub(USER_MENTION_REGEX, " [مستخدم] ", text)

    if remove_html_markup:
        # remove html line breaks
        text = re.sub("<br />", " ", text)
        # remove html markup
        text = re.sub("</?[^>]+>", " ", text)

    if map_hindi_numbers_to_arabic:
        text = text.translate(HINDI_TO_ARABIC_MAP)

    # remove repeated characters >2
    if remove_non_digit_repetition:
        text = _remove_non_digit_repetition(text)

    # insert whitespace before and after all non Arabic digits or English Digits and Alphabet and the 2 brackets
    if insert_white_spaces:
        text = re.sub(
            "([^0-9\u0621-\u063A\u0641-\u064A\u0660-\u0669a-zA-Z ])",
            r" \1 ",
            text,
        )

        # re-fix brackets
        text = text.replace("[ رابط ]", "[رابط]")
        text = text.replace("[ بريد ]", "[بريد]")
        text = text.replace("[ مستخدم ]", "[مستخدم]")

        # insert whitespace between words and numbers or numbers and words
        text = re.sub(
            "(\d+)([\u0621-\u063A\u0641-\u064A\u066A-\u066C\u0654-\u0655]+)",
            r" \1 \2 ",
            text,
        )
        text = re.sub(
            "([\u0621-\u063A\u0641-\u064A\u066A-\u066C\u0654-\u0655]+)(\d+)",
            r" \1 \2 ",
            text,
        )

    # remove unwanted characters
    text = re.sub(self_REJECTED_CHARS_REGEX, " ", text)

    # remove extra spaces
    text = " ".join(text.replace("\uFE0F", "").split())
    farasa_segmenter1 = FarasaSegmenter(interactive=True)
    if keep_emojis:
        new_text = []
        for word in text.split():
            if word in list(self_emoji.get_emoji_unicode_dict("en").keys()):
                new_text.append(word)
            else:
                new_text.append(farasa_segmenter1.segment(word))
        text = " ".join(new_text)
    norm_text = text

    if not norm_text.strip():
        norm_text = text1
        # raise ValueError("Text `%s` cannot be processed!!!" % text)
    norm_text = normalize_text(norm_text)
    norm_text = norm_text.strip()
    # ALl the other models dont require Farasa Segmentation
    return norm_text
# def _farasa_segment(text: str) -> str:
#     line_farasa = text.split()
#     segmented_line = []
#     for index, word in enumerate(line_farasa):
#         if word in ["[", "]"]:
#             continue
#         if word in ["رابط", "بريد", "مستخدم"] and line_farasa[index - 1] in [
#             "[",
#             "]",
#         ]:
#             segmented_line.append("[" + word + "]")
#             continue
#         if "+" not in word:
#             segmented_line.append(word)
#             continue
#         segmented_word = _split_farasa_output(word)
#         segmented_line.extend(segmented_word)

#     return " ".join(segmented_line)
# def _split_farasa_output(word: str) -> str:
#     segmented_word = []
#     temp_token = ""
#     for i, c in enumerate(word):
#         if c == "+":
#             # if the token is KAF, it could be a suffix or prefix
#             if temp_token == "ك":
#                 # if we are at the second token, then KAF is surely a prefix
#                 if i == 1:
#                     segmented_word.append(temp_token + "+")
#                     temp_token = ""
#                 # If the KAF token is between 2 tokens
#                 elif word[i - 2] == "+":
#                     # if the previous token is prefix, then this KAF must be a prefix
#                     if segmented_word[-1][-1] == "+":
#                         segmented_word.append(temp_token + "+")
#                         temp_token = ""
#                     # else it is a suffix, this KAF could not be a second suffix
#                     else:
#                         segmented_word.append("+" + temp_token)
#                         temp_token = ""
#                 # if Kaf is at the end, this is handled with the statement after the loop
#             elif temp_token in PREFIX_LIST:
#                 segmented_word.append(temp_token + "+")
#                 temp_token = ""
#             elif temp_token in SUFFIX_LIST:
#                 segmented_word.append("+" + temp_token)
#                 temp_token = ""
#             else:
#                 segmented_word.append(temp_token)
#                 temp_token = ""
#             continue
#         temp_token += c
#     if temp_token != "":
#         if temp_token in SUFFIX_LIST:
#             segmented_word.append("+" + temp_token)
#         else:
#             segmented_word.append(temp_token)
#     return segmented_word