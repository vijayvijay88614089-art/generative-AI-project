import unittest
from modules.rag_pipeline import Chunk, answer, retrieve


class RagTests(unittest.TestCase):
    def test_retrieves_relevant_passage(self):
        chunks = [Chunk("guide.txt", 1, "Revenue is calculated as unit price multiplied by units sold."),
                  Chunk("guide.txt", 2, "The customer segment field groups shoppers by use case.")]
        hits = retrieve(chunks, "How do we calculate revenue?")
        self.assertTrue(hits)
        self.assertIn("Revenue", hits[0]["chunk"].text)

    def test_empty_or_unsearchable_documents_return_no_hits(self):
        self.assertEqual(retrieve([], "question"), [])
        self.assertEqual(retrieve([Chunk("blank.txt", None, "... !!!")], "the and"), [])

    def test_character_fallback_catches_word_form_differences(self):
        chunks = [Chunk("guide.txt", 1, "The satisfaction score improved after the service update.")]
        hits = retrieve(chunks, "satisfactions")
        self.assertTrue(hits)
        self.assertIn("Character similarity", hits[0]["match_type"])

    def test_no_match_explains_retrieval_limits_without_inventing_answer(self):
        response, source = answer("Where is the office?", [])
        self.assertEqual(source, "No retrieval match")
        self.assertIn("No answer was generated", response)


if __name__ == "__main__": unittest.main()
