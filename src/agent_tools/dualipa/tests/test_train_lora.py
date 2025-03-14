"""
Tests for the train_lora module.

Official Documentation References:
- pytorch: https://pytorch.org/docs/stable/index.html
- transformers: https://huggingface.co/docs/transformers/index
- peft: https://huggingface.co/docs/peft/index
- pytest: https://docs.pytest.org/
- unittest.mock: https://docs.python.org/3/library/unittest.mock.html
- tempfile: https://docs.python.org/3/library/tempfile.html
"""

import pytest
import torch
from unittest.mock import patch, MagicMock
import os
import json
import tempfile
from pathlib import Path

# Import the function to be tested
from agent_tools.dualipa.train_lora import train_lora


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    data = {
        "question": "What does this function do?",
        "answer": "This function does something."
    }
    
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
        json.dump([data], temp_file)
        temp_file_path = temp_file.name
    
    yield temp_file_path
    
    # Cleanup
    if os.path.exists(temp_file_path):
        os.remove(temp_file_path)


@pytest.fixture
def output_dir():
    """Create a temporary output directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    
    # Cleanup - remove directory and its contents
    import shutil
    shutil.rmtree(temp_dir)


@patch("agent_tools.dualipa.train_lora.AutoTokenizer.from_pretrained")
@patch("agent_tools.dualipa.train_lora.AutoModelForCausalLM.from_pretrained")
@patch("agent_tools.dualipa.train_lora.get_peft_model")
@patch("agent_tools.dualipa.train_lora.load_dataset")
def test_train_lora_initialization(mock_load_dataset, mock_get_peft_model, 
                                   mock_model, mock_tokenizer, sample_dataset):
    """Test that train_lora initializes all the required components correctly."""
    # Setup mocks
    mock_dataset = MagicMock()
    mock_dataset.__iter__.return_value = [{"question": "test", "answer": "response"}]
    mock_load_dataset.return_value = mock_dataset
    
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    
    mock_peft_instance = MagicMock()
    mock_peft_instance.train.return_value = None
    mock_loss = MagicMock()
    mock_loss.backward.return_value = None
    mock_peft_instance.return_value = MagicMock(loss=mock_loss)
    mock_get_peft_model.return_value = mock_peft_instance
    
    # Call the function
    train_lora(sample_dataset, model_name="test-model")
    
    # Verify function calls
    mock_load_dataset.assert_called_once()
    mock_tokenizer.assert_called_once_with("test-model")
    mock_model.assert_called_once()
    mock_get_peft_model.assert_called_once()
    
    # Verify model was set to train mode
    mock_peft_instance.print_trainable_parameters.assert_called_once()


@patch("agent_tools.dualipa.train_lora.AutoTokenizer.from_pretrained")
@patch("agent_tools.dualipa.train_lora.AutoModelForCausalLM.from_pretrained")
@patch("agent_tools.dualipa.train_lora.get_peft_model")
@patch("agent_tools.dualipa.train_lora.load_dataset")
@patch("agent_tools.dualipa.train_lora.Trainer")
def test_train_lora_with_default_model(mock_trainer, mock_load_dataset, mock_get_peft_model, 
                                       mock_model, mock_tokenizer, sample_dataset, output_dir):
    """Test that train_lora uses the default model when none is specified."""
    # Setup mocks
    mock_dataset = MagicMock()
    mock_dataset.__iter__.return_value = [{"question": "test", "answer": "response"}]
    mock_load_dataset.return_value = mock_dataset
    
    mock_tokenizer_instance = MagicMock()
    mock_tokenizer_instance.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
    mock_tokenizer.return_value = mock_tokenizer_instance
    
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    
    mock_peft_instance = MagicMock()
    mock_peft_instance.train.return_value = None
    mock_loss = MagicMock()
    mock_loss.backward.return_value = None
    mock_peft_instance.return_value = MagicMock(loss=mock_loss)
    mock_get_peft_model.return_value = mock_peft_instance
    
    mock_trainer_instance = MagicMock()
    mock_trainer.return_value = mock_trainer_instance
    
    # Call the function with default model
    train_lora(sample_dataset, output_dir=output_dir)
    
    # Verify default model is used
    mock_tokenizer.assert_called_once_with("unsloth/Mistral-7B")
    mock_model.assert_called_once()
    
    # Verify trainer was created and train was called
    mock_trainer.assert_called_once()
    mock_trainer_instance.train.assert_called_once()


def test_train_lora_with_missing_dataset():
    """Test that train_lora raises FileNotFoundError for missing dataset."""
    with pytest.raises(FileNotFoundError):
        train_lora("nonexistent_dataset.json")


@patch("agent_tools.dualipa.train_lora.torch.cuda.is_available", return_value=False)
@patch("agent_tools.dualipa.train_lora.load_dataset")
@patch("agent_tools.dualipa.train_lora.AutoTokenizer.from_pretrained")
@patch("agent_tools.dualipa.train_lora.AutoModelForCausalLM.from_pretrained")
@patch("agent_tools.dualipa.train_lora.get_peft_model")
@patch("agent_tools.dualipa.train_lora.Trainer")
def test_train_lora_on_cpu(mock_trainer, mock_get_peft_model, mock_model, 
                          mock_tokenizer, mock_load_dataset, mock_cuda_check, 
                          sample_dataset, output_dir):
    """Test that train_lora works correctly on CPU when CUDA is not available."""
    # Setup mocks
    mock_load_dataset.return_value = MagicMock()
    mock_tokenizer.return_value = MagicMock()
    mock_model.return_value = MagicMock()
    mock_get_peft_model.return_value = MagicMock()
    mock_trainer.return_value = MagicMock()
    
    # Call the function
    train_lora(sample_dataset, output_dir=output_dir)
    
    # Verify model was loaded with float32 when on CPU
    mock_model.assert_called_once_with("unsloth/Mistral-7B", torch_dtype=torch.float32)
    
    # Verify training args were created with fp16=False
    trainer_args = mock_trainer.call_args[1]
    assert "args" in trainer_args
    assert not trainer_args["args"].fp16 