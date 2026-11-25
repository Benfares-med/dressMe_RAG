import unittest

from src.domain_guard import is_fashion_question


class DomainGuardTests(unittest.TestCase):
    def test_accepts_fashion_questions(self):
        questions = [
            "Qu'est-ce que le style Old Money ?",
            "Propose une tenue pour un mariage en hiver.",
            "Quelles couleurs porter avec un blazer ?",
            "Et pour l'été ?",
            "Comment dois-je m'habiller demain ?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertTrue(is_fashion_question(question))

    def test_rejects_general_knowledge_questions(self):
        questions = [
            "Quelle est la capitale du Maroc ?",
            "Combien font 2 + 2 ?",
            "Qui a écrit Les Misérables ?",
        ]
        for question in questions:
            with self.subTest(question=question):
                self.assertFalse(is_fashion_question(question))


if __name__ == "__main__":
    unittest.main()