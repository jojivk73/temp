

from datasets import load_dataset
import transformers
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers import TFAutoModelForCausalLM
import numpy as np;
from typing import Dict, Optional, Sequence
#from add_tokens import add_tokens, resize_model_embeddings
from tensorflow.keras.losses import SparseCategoricalCrossentropy

model_name = "gpt2-xl"
raw_datasets = load_dataset("tatsu-lab/alpaca")
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(example):
  return tokenizer(text=example["text"], text_target=example["output"], padding="longest", return_tensors="np", truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="np")

tf_train_dataset = tokenized_datasets["train"].to_tf_dataset(
   columns=["attention_mask", "input_ids"],
   label_cols=["labels"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
model = TFAutoModelForCausalLM.from_pretrained(model_name)
from tensorflow.keras.optimizers.schedules import PolynomialDecay
num_epochs=3
num_train_steps = len(tf_train_dataset) * num_epochs
lr_scheduler = PolynomialDecay(
    initial_learning_rate=2e-5, end_learning_rate=0.0, decay_steps=num_train_steps
)
from tensorflow.keras.optimizers import Adam

opt = Adam(learning_rate=lr_scheduler)

model.compile(
    optimizer=opt,
    loss=SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
model.fit(
    tf_train_dataset,
    epochs=num_epochs
)
