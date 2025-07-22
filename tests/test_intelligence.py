import unittest
from src.agents.preference_feedback_loop import PreferenceFeedbackLoopAgent
from src.agents.predictive_intent import PredictiveIntentAgent
from src.agents.art_vision import ArtVisionAgent

class TestIntelligence(unittest.TestCase):
    def test_relevance(self):
        # Simulate 50 interactions
        feedback = PreferenceFeedbackLoopAgent()
        predictive = PredictiveIntentAgent()
        art = ArtVisionAgent()
        relevance_count = 0
        for i in range(50):
            # Simulate
            feedback._process({"style": "vector"})
            pred = predictive._process({})
            sug = art._process({"user_style": "vector"})
            if "vector" in sug["suggestion"]:  # simplistic check
                relevance_count += 1
        self.assertGreaterEqual(relevance_count / 50, 0.95)

if __name__ == "__main__":
    unittest.main()
