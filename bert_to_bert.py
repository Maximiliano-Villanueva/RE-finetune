# Standard imports
import logging
from typing import Dict, List

# Third-party imports
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (BertTokenizer, EncoderDecoderModel,
                          Seq2SeqTrainer, Seq2SeqTrainingArguments)

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def compute_metrics(pred) -> Dict[str, float]:
    """
    Function to compute token-wise accuracy.
    Args:
    - pred: Predicted outputs
    
    Returns:
    - A dictionary containing the accuracy metric.
    """
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    labels_flat = labels.flatten()
    preds_flat = preds.flatten()
    accuracy = accuracy_score(labels_flat, preds_flat)
    return {"accuracy": accuracy}

# Standard imports
from typing import Union, List

# Third party imports
from transformers import BertTokenizer
from datasets import Dataset

import logging

# Initialize the logger
logger = logging.getLogger(__name__)

def create_dataset_repeats(
    input_text: Union[str, List[str]],
    output_text: Union[str, List[str]],
    repeats: int = 100
) -> Dataset:
    """
    Creates a dataset with repeated samples.
    
    Args:
    - input_text (Union[str, List[str]]): Original input text or list of input texts.
    - output_text (Union[str, List[str]]): Original output text or list of output texts.
    - repeats (int, optional): Number of times the sample should be repeated. Defaults to 100.
    
    Returns:
    - Dataset: A Dataset object with repeated samples.
    """
    logger.info("Creating a dataset with repeated samples.")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    # Check if input_text and output_text are strings and convert them to list if they are
    if isinstance(input_text, str):
        input_text = [input_text] * repeats
    else:
        input_text = input_text * repeats
    if isinstance(output_text, str):
        output_text = [output_text] * repeats
    else:
        output_text = output_text * repeats

    input_tokenized = tokenizer(input_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    output_tokenized = tokenizer(output_text, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    
    logger.info(f"Length of input_tokenized.input_ids: {len(input_tokenized.input_ids)}")
    logger.info(f"Length of output_tokenized.input_ids: {len(output_tokenized.input_ids)}")
   

    return Dataset.from_dict({
        "input_ids": input_tokenized.input_ids, 
        "attention_mask": input_tokenized.attention_mask, 
        "labels": output_tokenized.input_ids
    })


# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-multilingual-cased", "bert-base-multilingual-cased")
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id

text1 = "La economía circular se ha convertido en una tendencia prometedora y vital. Alberto Rodríguez ha estado trabajando con baterías eléctricas desde el año 2010"
output1 = "La economía circular@@@se ha convertido en@@@una tendencia prometedora y vital |||  Alberto Rodríguez@@ha estado trabajando con@@baterías eléctricas desde el año 2010"

text2= """se reciclan baterías usadas desde el año 2013. planta de Nissan reciclan baterías usadas
Carlos Ghosn habló la visión de la empresa de reducir el impacto negativo de sus productos en el medio ambiente. Carlos Ghosn Destacó el uso de baterías eléctricas es fundamental para reducir significativamente las emisiones de CO2
su equipo profesional se esfuerza en desarrollar tecnologías más avanzadas. María Fernández Nacida en 1987 en Barcelona
La economía circular de las baterías eléctricas es beneficiosa para el medio ambiente. Otra empresa trabaja en la economía circular de las baterías eléctricas
Umicore es especialista en la fabricación y reciclaje de baterías. Umicore abrió una planta de reciclaje de baterías de iones de litio en Bélgica en 2019"""

output2="""se reciclan@@@baterías usadas@@@desde el año 2013. planta de Nissan@@@reciclan@@@baterías usadas
Carlos Ghosn@@@habló@@@la visión de la empresa de reducir el impacto negativo de sus productos en el medio ambiente. Carlos Ghosn@@@Destacó@@@el uso de baterías eléctricas es fundamental para reducir significativamente las emisiones de CO2
su equipo profesional@@@se esfuerza en desarrollar@@@tecnologías más avanzadas. María Fernández@@@Nacida en@@@1987 en Barcelona
La economía circular de las baterías eléctricas@@@es@@@beneficiosa para el medio ambiente. Otra empresa@@@trabaja@@@en la economía circular de las baterías eléctricas
Umicore@@@es especialista en@@@la fabricación y reciclaje de baterías. Umicore@@@abrió@@@una planta de reciclaje de baterías de iones de litio en Bélgica en 2019"""

# Prepare data with 100 repeated samples
train_data = create_dataset_repeats(
    input_text=text2.split("\n"),
    output_text=output2.split("\n"),
    repeats=100
)

# Training arguments
training_args = Seq2SeqTrainingArguments(
    per_device_train_batch_size=1,
    save_total_limit=2,
    output_dir="./",
    overwrite_output_dir=True,
    num_train_epochs=10,
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    args=training_args,
    train_dataset=train_data,
    compute_metrics=compute_metrics,
)

# Log the start of training
logger.info("Starting training...")

# Train (This should overfit the model to the single example)
trainer.train()

# Log the end of training
logger.info("Training completed.")

# Evaluate the model
bert2bert.eval()

with torch.no_grad():
    example_input = "La economía circular se ha convertido en una tendencia prometedora y vital. Alberto Rodríguez ha estado trabajando con baterías eléctricas desde el año 2010"
    input_tokenized = tokenizer(example_input, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    outputs = bert2bert.generate(input_tokenized.input_ids, attention_mask=input_tokenized.attention_mask)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

print(f"Predicted Output: {output_str}")
