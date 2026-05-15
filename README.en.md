# Automatic-Romanian-Text-Generation-using-GPT-2

[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717)](https://github.com/MariusCristianBuzea/Automatic-Romanian-Text-Generation-using-GPT-2)
![Language](https://img.shields.io/badge/Language-Romanian%20NLP-0b7285)
![Python](https://img.shields.io/badge/Python-3.x-3776ab)
![Transformers](https://img.shields.io/badge/Transformers-Hugging%20Face-ffcc00)

## Overview

Automatic Romanian text generation with GPT-2 and online news evaluation.

The project focuses on **named entity recognition** for Romanian language data. This English README keeps the technical content from the Romanian documentation in English form: model links, dataset notes, reported results, setup commands, Docker/API usage and relevant files.

## Data and Results

Key dataset sizes, metrics or training values mentioned in the original README:
- ROUGE-1
- BLEURT checkpoint was used – a self-contained folder that contained a regression model which was tested on several languages, but should work for the 100+ languages of multilingual C4 (a cleaned version of Common Crawl’s web crawl corpus), including the Romanian language with 45M of training and 45
- BLEURT-20 was used as checkpoint, being a 32 layers pre-trained transformer model, named RemBERT, which contained 579M parameters fine-tuned on human ratings and synthetic data (~590K sentence pairs) collected during years 2015 and 2019
- F1 = score([text], [new_text], model_type=output_dir, num_layers=9,
- BLEU metric values of generated news items (30 values) using RoGPT-2 and MCBGPT-2
- ROUGE (red), BLEURT (green) and BERTScore (gray) metrics values (30
- BLEU, ROUGE, BLEURT and BERTScore, the MCBGPT-2 and RoGPT-2
- BERTScore metric when are evaluated larger sentences (Table 4
- 536 articles
- 14,756 news items
- 4922 news items
- corpus, that contain 18 fake news and 24,582 true news, 368 news with negative polarity and 2135 news with positive polarity, being split in equal parts between train, test and validation datasets

Important tables preserved from the original documentation:

```text
|                               |    Test     | 957,787    |                               |    Test     |    194     |
|    Romanian unique words      | Validation  | 1,026,680  |  Average words  per news item | Validation  |    208     |
|                               |   Train     | 2,837,236  |                               |    Train    |    192     |
```

```text
| Number of parameters |       131M         | 
|    Number of epoch   |        15          |  
| Duration of an epoch |        5h          | 
|     Context size     |       512          | 
|      Batch size      |        12          | 
```

```text
|       RoGPT-2        |       Test         |    32.29  |   0.53  |  0.68  |  0.8106   | 
|                      |    Validation      |    28.70  |   0.50  |  0.52  |  0.8136   | 
|       MCBGPT-2       |       Test         |    8.79   |   0.25  |  0.63  |  0.8124   | 
|                      |    Validation      |    9.11   |   0.14  |  0.50  |  0.8277   | 
```


## Setup and Usage

Relevant commands and examples preserved from the original README:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
```python
block_size = 512
BATCH_SIZE = 12
BUFFER_SIZE = 1000
# defining our optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08, clipnorm=1.0)
# definining our loss function
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# defining our metric which we want to observe
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
# compiling the model
model.compile(optimizer=optimizer, loss=[loss, *[None] * model.config.n_layer], metrics=[metric])
num_epoch = 15
history = model.fit(dataset, epochs=num_epoch)
from transformers import WEIGHTS_NAME, CONFIG_NAME
output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/mcb_model"
# creating directory if it is not present
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_to_save = model.module if hasattr(model, 'module') else model
output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
output_config_file = os.path.join(output_dir, CONFIG_NAME)
# save model and model configs
model.save_pretrained(output_dir)
model_to_save.config.to_json_file(output_config_file)
# save tokenizer
tokenizer.save_pretrained(output_dir)
```
```python
output_dir = configuration.get_property('PYTHON_DIR') + "/gpt_models/mcb_model/"
CUDA_LAUNCH_BLOCKING = 1
# TensorFlow
from transformers import AutoTokenizer, TFAutoModelForCausalLM

results = sqlI.select_all_rows_from_table(['columns'], 'table', 'database', None, 'primaryKey DESC')
tokenizer = AutoTokenizer.from_pretrained(output_dir)
model = TFAutoModelForCausalLM.from_pretrained(output_dir)

updated_texts = dict()
for result in results:
    updated_texts.update({result[0]: configuration.clean_text(result[1])})

for id_single, text in updated_texts.items():
    max_length = 4000
    min_length = 600
    inputs = tokenizer.encode(text, return_tensors='tf', max_length=512, truncation=True)
    text_predicted = model.generate(inputs, max_length=int(max_length), min_length=int(min_length),
                                    no_repeat_ngram_size=2,
                                    temperature=0.8, num_beams=5, num_return_sequences=5, )
    new_text = tokenizer.decode(text_predicted[0], skip_special_tokens=True).replace('<|endoftext|>', '')
    new_text = new_text.replace("'", ' ')
    sqlI.update_field_for_all_rows_from_table(['column'], "'" + new_text + "'",
                                              'table',
                                              'primaryKey="' + str(id_single) + '" AND primaryKey > 0',
                                              'database').
```

## Relevant Files

- `requirements.txt`

## Notes

- Romanian sample texts may appear inside code blocks because the models are trained and evaluated on Romanian language tasks.
- The Romanian `README.md` remains available as the original documentation.
- Absolute local filesystem paths should not be used; prefer relative paths such as `models/model`, `data/input.json` or paths mounted inside Docker containers.
