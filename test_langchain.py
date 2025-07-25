#!/usr/bin/env python3
"""
Simple LangChain test script
"""

from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from langchain.chains import LLMChain
from langchain.llms.fake import FakeListLLM

class SimpleOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a simple string."""
    
    def parse(self, text: str) -> str:
        """Parse the output of an LLM call."""
        return text.strip()

def test_langchain_basic():
    """Test basic LangChain functionality without requiring API keys"""
    
    print("Testing LangChain Basic Functionality")
    print("=" * 40)
    
    # Create a fake LLM for testing (doesn't require API keys)
    responses = [
        "Artificial Intelligence is a field of computer science that aims to create intelligent machines.",
        "Machine Learning is a subset of AI that enables computers to learn without being explicitly programmed.",
        "Deep Learning uses neural networks with multiple layers to process data."
    ]
    
    fake_llm = FakeListLLM(responses=responses)
    
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Explain {topic} in simple terms:"
    )
    
    # Create a chain
    chain = LLMChain(
        llm=fake_llm,
        prompt=prompt,
        output_parser=SimpleOutputParser()
    )
    
    # Test topics
    topics = ["artificial intelligence", "machine learning", "deep learning"]
    
    for topic in topics:
        print(f"\nTopic: {topic}")
        result = chain.run(topic=topic)
        print(f"Response: {result}")
    
    print("\n" + "=" * 40)
    print("LangChain test completed successfully!")

if __name__ == "__main__":
    test_langchain_basic()