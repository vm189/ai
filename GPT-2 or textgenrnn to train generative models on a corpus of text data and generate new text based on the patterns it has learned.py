import os
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments

# Step 1: Load the Dataset
print("Loading dataset...")
dataset = load_dataset("tiny_shakespeare")

# Split into train and test sets
train_data = dataset["train"]
test_data = dataset["test"]

# Step 2: Load Pre-trained GPT-2 Tokenizer and Model
print("Loading GPT-2 tokenizer and model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 3: Tokenize the Dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)

print("Tokenizing dataset...")
tokenized_train = train_data.map(tokenize_function, batched=True)
tokenized_test = test_data.map(tokenize_function, batched=True)
# Step 4: Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_total_limit=2,
    logging_dir="./logs",
    logging_steps=10,
)

# Step 5: Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

# Step 6: Train the Model
print("Starting training...")
trainer.train()

# Step 7: Save the Fine-Tuned Model
model.save_pretrained("./fine_tuned_gpt2")
tokenizer.save_pretrained("./fine_tuned_gpt2")
print("Model saved to './fine_tuned_gpt2'.")

# Step 8: Generate Text Using the Fine-Tuned Model
print("Generating new text...")
model.eval()
input_text = "To be or not to be, that is the"
inputs = tokenizer.encode(input_text, return_tensors="pt")
outputs = model.generate(inputs, max_length=100, num_return_sequences=1, temperature=0.7)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nGenerated Text:\n")
print(generated_text)
