"""
Train a LoRA model using the extracted dataset.

Official Documentation References:
- PyTorch: https://pytorch.org/docs/stable/index.html
- Transformers: https://huggingface.co/docs/transformers/index
- PEFT (LoRA): https://huggingface.co/docs/peft/index
- Unsloth: https://github.com/unslothai/unsloth
- Datasets: https://huggingface.co/docs/datasets/index
"""

import os
import torch
import unsloth
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from loguru import logger
import json

def train_lora(dataset_path: str, 
               output_dir: str = "./models", 
               model_name: str = "unsloth/Mistral-7B",
               r: int = 16,
               lora_alpha: int = 32,
               lora_dropout: float = 0.1,
               learning_rate: float = 2e-4,
               num_train_epochs: int = 3,
               max_steps: int = -1,
               per_device_train_batch_size: int = 1) -> None:
    """Trains a LoRA model using the extracted dataset.
    
    Args:
        dataset_path: Path to the JSON dataset containing question-answer pairs
        output_dir: Directory where trained model will be saved
        model_name: Base model to use for fine-tuning
        r: LoRA attention dimension
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate
        learning_rate: Learning rate for training
        num_train_epochs: Number of training epochs
        max_steps: Maximum number of training steps (-1 to use epoch setting)
        per_device_train_batch_size: Batch size per device during training
        
    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        RuntimeError: If CUDA is requested but not available
    """
    try:
        # Validate dataset path
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
        logger.info(f"Loading dataset from {dataset_path}")
        # Load dataset
        dataset = load_dataset("json", data_files=dataset_path, split="train")
        logger.info(f"Loaded dataset with {len(dataset)} examples")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if CUDA is available when using PyTorch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        if device == "cpu":
            logger.warning("Training on CPU is very slow. Consider using a machine with a GPU.")

        # Load tokenizer and model
        logger.info(f"Loading tokenizer and model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

        # Configure LoRA
        logger.info(f"Configuring LoRA with r={r}, alpha={lora_alpha}, dropout={lora_dropout}")
        target_modules = ["q_proj", "v_proj"]
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules
        )

        # Apply LoRA to the model
        peft_model = get_peft_model(model, lora_config)
        peft_model.print_trainable_parameters()

        # Tokenize the dataset
        def tokenize_function(examples):
            # Tokenize questions and answers separately
            question_encodings = tokenizer(examples["question"], truncation=True, padding="max_length", max_length=512)
            answer_encodings = tokenizer(examples["answer"], truncation=True, padding="max_length", max_length=512)
            
            # Combine them for training
            encodings = {
                "input_ids": question_encodings["input_ids"],
                "attention_mask": question_encodings["attention_mask"],
                "labels": answer_encodings["input_ids"]
            }
            return encodings
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_steps=100,
            save_total_limit=3,
            fp16=device == "cuda"  # Use mixed precision only with CUDA
        )

        # Create Trainer
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset,
            tokenizer=tokenizer
        )

        # Train the model
        logger.info("Starting training")
        trainer.train()
        
        # Save the model
        logger.info(f"Training completed. Saving model to {output_dir}")
        peft_model.save_pretrained(os.path.join(output_dir, "lora_adapter"))
        tokenizer.save_pretrained(os.path.join(output_dir, "tokenizer"))

        print(f"Training completed and model saved to {output_dir}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        raise
    except RuntimeError as e:
        logger.error(f"Runtime error during training: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a LoRA model")
    parser.add_argument("dataset_path", help="Path to the dataset JSON file")
    parser.add_argument("--output-dir", default="./models", help="Output directory for trained model")
    parser.add_argument("--model-name", default="unsloth/Mistral-7B", help="Base model to use")
    parser.add_argument("--r", type=int, default=16, help="LoRA attention dimension")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha parameter")
    parser.add_argument("--lora-dropout", type=float, default=0.1, help="LoRA dropout rate")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per device")
    
    args = parser.parse_args()
    
    train_lora(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        r=args.r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size
    )
