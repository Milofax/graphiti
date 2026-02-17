"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

STOPWORDS = [
    # === ENGLISH ===
    # Articles & determiners
    'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some', 'any', 'each', 'every',
    # Pronouns
    'i', 'me', 'my', 'mine', 'myself',
    'you', 'your', 'yours', 'yourself',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    # Be/have/do verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'has', 'have', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done',
    # Modal verbs
    'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must',
    # Common verbs
    'get', 'gets', 'got', 'getting',
    'go', 'goes', 'went', 'going', 'gone',
    'make', 'makes', 'made', 'making',
    'come', 'comes', 'came', 'coming',
    'take', 'takes', 'took', 'taking', 'taken',
    'give', 'gives', 'gave', 'given',
    'say', 'says', 'said',
    'know', 'knows', 'knew', 'known',
    'see', 'sees', 'saw', 'seen',
    'use', 'uses', 'used', 'using',
    'find', 'finds', 'found',
    'want', 'need', 'keep', 'let', 'put', 'set', 'run', 'show',
    # Prepositions & conjunctions
    'at', 'by', 'for', 'from', 'in', 'into', 'of', 'on', 'to', 'with',
    'about', 'after', 'before', 'between', 'through', 'during', 'without',
    'above', 'below', 'under', 'over', 'up', 'down', 'out', 'off',
    'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
    'if', 'then', 'than', 'when', 'where', 'while', 'because', 'since', 'until', 'although',
    # Adverbs
    'not', 'no', 'also', 'very', 'just', 'only', 'even', 'still', 'already',
    'always', 'never', 'often', 'usually', 'generally', 'sometimes',
    'here', 'there', 'now', 'again', 'once',
    'how', 'what', 'which', 'who', 'whom', 'whose', 'why',
    # Other common
    'own', 'other', 'another', 'such', 'much', 'many', 'more', 'most',
    'all', 'few', 'less', 'same', 'well', 'back',
    'new', 'old', 'first', 'last', 'long', 'great', 'little', 'right',
    'big', 'high', 'different', 'small', 'large', 'next', 'early',
    'able', 'like', 'however', 'way', 'thing', 'things',
    'as',
    # === DEUTSCH ===
    # Artikel & Determinanten
    'der', 'die', 'das', 'den', 'dem', 'des',
    'ein', 'eine', 'einer', 'einem', 'einen', 'eines',
    'kein', 'keine', 'keiner', 'keinem', 'keinen', 'keines',
    # Pronomen
    'ich', 'du', 'er', 'sie', 'es', 'wir', 'ihr',
    'mich', 'dich', 'sich', 'uns', 'euch',
    'mir', 'dir', 'ihm', 'ihnen',
    'mein', 'dein', 'sein', 'unser', 'euer',
    'dieser', 'diese', 'dieses', 'jener', 'jene', 'jenes',
    'man', 'was', 'wer', 'welche', 'welcher', 'welches',
    # Verben (sein/haben/werden)
    'ist', 'bin', 'bist', 'sind', 'seid', 'war', 'waren', 'gewesen',
    'hat', 'habe', 'hast', 'haben', 'habt', 'hatte', 'hatten',
    'wird', 'werde', 'wirst', 'werden', 'werdet', 'wurde', 'wurden',
    # Modalverben
    'kann', 'kannst', 'muss', 'soll', 'darf', 'mag', 'will',
    # Präpositionen & Konjunktionen
    'auf', 'aus', 'bei', 'bis', 'durch', 'nach', 'ohne', 'um', 'unter',
    'vor', 'zwischen', 'gegen', 'seit', 'von', 'zu', 'zum', 'zur',
    'mit', 'als', 'wie', 'ob', 'dass', 'weil', 'wenn', 'aber', 'oder',
    'und', 'denn', 'sondern', 'nicht', 'noch', 'schon', 'auch', 'nur',
    'so', 'da', 'dann', 'doch', 'sehr', 'immer', 'hier', 'dort',
    # Andere häufige
    'andere', 'anderer', 'anderes', 'anderen',
    'alle', 'alles', 'allem', 'allen', 'aller',
    'viel', 'viele', 'mehr', 'ganz', 'etwa', 'dabei',
]

# Safety net: cap OR-terms to prevent fulltext explosion on long queries.
# RediSearch OR-queries (term1 | term2 | ... | termN) scale linearly with term count —
# each term traverses a separate posting list and all matches get BM25-scored.
MAX_FULLTEXT_TERMS = 5
