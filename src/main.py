from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOTrainer, GRPOConfig
import torch

def dummy_reward(prompts, completions, **kwargs):
    """
    Returns a constant reward for every completion.
    
    Args:
        prompts: List of prompt texts
        completions: List of generated completion texts
        **kwargs: Additional arguments passed by the trainer
    
    Returns:
        List of reward scores (one per completion)
    """
    return [1.0 for _ in range(len(completions))]

def load_data():
    dataset = load_dataset("imdb", split="train[:1%]")
    dataset = dataset.map(lambda x: {"prompt": x["text"][:512]}, remove_columns=dataset.column_names)
    return dataset

def load_model():
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def train():
    dataset = load_data()
    tokenizer, model = load_model()

    config = GRPOConfig(
        output_dir="./checkpoints",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        max_steps=5,
        logging_steps=1,
        num_generations=2,
        bf16=False,
        fp16=False
    )

    trainer = GRPOTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        reward_funcs=[dummy_reward],
    )

    trainer.train()
    trainer.save_model("./checkpoints/final")

if __name__ == "__main__":
    train()