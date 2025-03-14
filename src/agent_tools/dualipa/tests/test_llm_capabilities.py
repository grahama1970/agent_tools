"""
Tests for basic LLM capabilities.

Official Documentation References:
- PyTorch: https://pytorch.org/docs/stable/index.html
- Transformers: https://huggingface.co/docs/transformers/index
- asyncio: https://docs.python.org/3/library/asyncio.html
"""

import pytest
import asyncio
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM


@pytest.mark.asyncio
async def test_text_generation_asyncio():
    """Test basic text generation using asyncio."""
    # Define a simple text generation function
    async def generate_text(prompt):
        # Use a small model for fast testing
        model_name = "distilgpt2"  # Much smaller than full GPT-2
        
        # Run the model in a separate thread to avoid blocking the event loop
        def _generate():
            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            
            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt")
            
            # Generate text (with minimal output for speed)
            with torch.no_grad():
                output = model.generate(
                    inputs.input_ids,
                    max_length=30,
                    num_return_sequences=1,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Decode the output
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            return generated_text
        
        # Run in a thread pool to avoid blocking
        return await asyncio.to_thread(_generate)
    
    # Generate a simple response
    prompt = "What is a large language model?"
    response = await generate_text(prompt)
    
    # Basic validation
    assert isinstance(response, str)
    assert len(response) > len(prompt)
    assert prompt in response  # The output should contain the prompt


def test_question_answering():
    """Test basic question answering capabilities."""
    # Use a small model for QA testing
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    # Test with a simple context and question
    context = "PyTorch is an open source machine learning framework. It was developed by Facebook."
    question = "Who developed PyTorch?"
    
    # Get the answer
    result = qa_pipeline(question=question, context=context)
    
    # Validate the result
    assert "score" in result
    assert "answer" in result
    assert "Facebook" in result["answer"]


def test_sentiment_analysis():
    """Test basic sentiment analysis capabilities."""
    # Use a small sentiment analysis model
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    
    # Test with positive and negative sentences
    positive_text = "I love this product, it's amazing!"
    negative_text = "This is the worst experience ever."
    
    # Get sentiment predictions
    positive_result = sentiment_pipeline(positive_text)[0]
    negative_result = sentiment_pipeline(negative_text)[0]
    
    # Validate the results
    assert positive_result["label"].lower() in ["positive", "pos"]
    assert negative_result["label"].lower() in ["negative", "neg"]
    assert positive_result["score"] > 0.5
    assert negative_result["score"] > 0.5


@pytest.mark.asyncio
async def test_parallel_inference():
    """Test running multiple model inferences in parallel with asyncio."""
    # Define a simple function to run a small model
    async def analyze_sentiment(text):
        def _analyze():
            sentiment_pipeline = pipeline(
                "sentiment-analysis", 
                model="distilbert-base-uncased-finetuned-sst-2-english"
            )
            return sentiment_pipeline(text)[0]
        
        return await asyncio.to_thread(_analyze)
    
    # Run multiple inferences in parallel
    texts = [
        "I'm having a great day!",
        "This is terrible news.",
        "The movie was fantastic.",
        "I'm really disappointed with the service."
    ]
    
    # Gather all results
    results = await asyncio.gather(*[analyze_sentiment(text) for text in texts])
    
    # Validate results
    assert len(results) == 4
    assert all("label" in result for result in results)
    assert all("score" in result for result in results)
    assert results[0]["label"].lower() in ["positive", "pos"]  # "I'm having a great day!"
    assert results[1]["label"].lower() in ["negative", "neg"]  # "This is terrible news." 