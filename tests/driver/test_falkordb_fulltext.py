"""Tests for FalkorDB fulltext query optimization.

Verifies:
1. STOPWORDS includes German and English common words
2. MAX_FULLTEXT_TERMS caps OR-terms to prevent fulltext explosion
3. build_fulltext_query correctly filters and caps terms
"""

import unittest
from unittest.mock import MagicMock, patch

import pytest

try:
    from graphiti_core.driver.falkordb_driver import (
        MAX_FULLTEXT_TERMS,
        STOPWORDS,
        FalkorDriver,
    )

    HAS_FALKORDB = True
except ImportError:
    HAS_FALKORDB = False


@unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
class TestStopwordsExpansion:
    """STOPWORDS must include common German and English words."""

    def test_stopwords_includes_german_articles(self):
        """German articles must be in STOPWORDS."""
        for word in ['der', 'die', 'das', 'den', 'dem', 'des', 'ein', 'eine']:
            assert word in STOPWORDS, f'German article "{word}" missing from STOPWORDS'

    def test_stopwords_includes_german_verbs(self):
        """German auxiliary/modal verbs must be in STOPWORDS."""
        for word in ['ist', 'hat', 'wird', 'kann', 'muss', 'sind', 'haben', 'werden']:
            assert word in STOPWORDS, f'German verb "{word}" missing from STOPWORDS'

    def test_stopwords_includes_german_prepositions(self):
        """German prepositions must be in STOPWORDS."""
        for word in ['auf', 'aus', 'bei', 'mit', 'nach', 'von', 'zu', 'und', 'oder']:
            assert word in STOPWORDS, f'German preposition "{word}" missing from STOPWORDS'

    def test_stopwords_includes_english_pronouns(self):
        """English pronouns must be in STOPWORDS."""
        for word in ['its', 'own', 'his', 'her', 'my', 'your', 'our', 'we', 'us']:
            assert word in STOPWORDS, f'English pronoun "{word}" missing from STOPWORDS'

    def test_stopwords_includes_common_verbs(self):
        """Common English verbs must be in STOPWORDS."""
        for word in ['goes', 'has', 'does', 'gets', 'made', 'went', 'got', 'came']:
            assert word in STOPWORDS, f'English verb "{word}" missing from STOPWORDS'

    def test_stopwords_includes_adverbs(self):
        """Common English adverbs must be in STOPWORDS."""
        for word in ['generally', 'usually', 'also', 'very', 'just', 'only']:
            assert word in STOPWORDS, f'English adverb "{word}" missing from STOPWORDS'

    def test_stopwords_case_insensitive_usage(self):
        """STOPWORDS list should contain lowercase entries only."""
        for word in STOPWORDS:
            assert word == word.lower(), f'STOPWORDS entry "{word}" is not lowercase'


@unittest.skipIf(not HAS_FALKORDB, 'FalkorDB is not installed')
class TestMaxFulltextTerms:
    """MAX_FULLTEXT_TERMS constant must exist and cap OR-terms."""

    def test_max_fulltext_terms_constant(self):
        """MAX_FULLTEXT_TERMS must be 10."""
        assert MAX_FULLTEXT_TERMS == 10

    def setup_method(self):
        """Set up FalkorDriver for testing."""
        with patch('graphiti_core.driver.falkordb_driver.FalkorDB'):
            self.driver = FalkorDriver()

    def test_build_fulltext_query_caps_terms(self):
        """Query with 20 non-stopwords produces max 10 OR-terms."""
        # 20 unique non-stopword terms
        words = [
            'alpha', 'bravo', 'charlie', 'delta', 'echo',
            'foxtrot', 'golf', 'hotel', 'india', 'juliet',
            'kilo', 'lima', 'mike', 'november', 'oscar',
            'papa', 'quebec', 'romeo', 'sierra', 'tango',
        ]
        query = ' '.join(words)
        result = self.driver.build_fulltext_query(query)
        terms = [t.strip() for t in result.split('|')]
        assert len(terms) <= MAX_FULLTEXT_TERMS, (
            f'Expected max {MAX_FULLTEXT_TERMS} terms, got {len(terms)}'
        )

    def test_build_fulltext_query_short_query_unchanged(self):
        """Query with 3 non-stopword terms keeps all 3."""
        query = 'Graphiti knowledge graph'
        result = self.driver.build_fulltext_query(query)
        terms = [t.strip() for t in result.split('|')]
        assert len(terms) == 3

    def test_build_fulltext_query_german_stopwords_filtered(self):
        """German stopwords are filtered from the query."""
        query = 'der Benutzer hat eine Anfrage'
        result = self.driver.build_fulltext_query(query)
        terms = [t.strip() for t in result.split('|')]
        # 'der', 'hat', 'eine' should be filtered; 'Benutzer', 'Anfrage' remain
        assert 'Benutzer' in terms
        assert 'Anfrage' in terms
        assert len(terms) == 2

    def test_build_fulltext_query_mixed_language(self):
        """Mixed DE/EN input filters stopwords from both languages."""
        query = 'the user goes into his own domain and der content wird generally used'
        result = self.driver.build_fulltext_query(query)
        terms = [t.strip() for t in result.split('|')]
        # Most words are stopwords: the, user(?), goes, into, his, own, and, der, wird, generally
        # 'user' is NOT a stopword, 'domain' is NOT, 'content' is NOT, 'used' IS
        # Verify none of the known stopwords appear
        for stopword in ['the', 'goes', 'into', 'his', 'own', 'and', 'der', 'wird', 'generally']:
            assert stopword not in terms, f'Stopword "{stopword}" should be filtered'

    def test_build_fulltext_query_timeout_query_from_logs(self):
        """The exact query that caused 60s timeouts must produce <= 10 terms."""
        # Real query from production logs (2026-02-14)
        query = (
            'requirement consciously choose Group ID follows rule domain '
            'specific content goes its own group generally transferable '
            'content goes main'
        )
        result = self.driver.build_fulltext_query(query)
        if result:  # Could be empty if all terms are stopwords
            terms = [t.strip() for t in result.split('|')]
            assert len(terms) <= MAX_FULLTEXT_TERMS, (
                f'Timeout query still produces {len(terms)} terms (max {MAX_FULLTEXT_TERMS})'
            )
