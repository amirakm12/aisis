#!/usr/bin/env python3
"""
Basic LangChain example
"""

import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Load environment variables
load_dotenv()

def main():
    """Main function demonstrating basic LangChain usage"""
    
    # Initialize the LLM (you'll need to set OPENAI_API_KEY in .env file)
    llm = OpenAI(temperature=0.7)
    
    # Create a prompt template
    prompt = PromptTemplate(
        input_variables=["topic"],
        template="Write a short explanation about {topic}:"
    )
    
    # Create a chain
    chain = LLMChain(llm=llm, prompt=prompt)
    
    # Example usage
    topic = "artificial intelligence"
    result = chain.run(topic=topic)
    
    print(f"Topic: {topic}")
    print(f"Response: {result}")

if __name__ == "__main__":
    main()