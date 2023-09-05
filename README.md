# RobustEmbed
a self-supervised sentence embedding framework that enhances both generalization and robustness benchmarks


#### Train the RobustEmbed embeddings to generate robust text represnetation
```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port $(expr $RANDOM + 1000) train2.py \
    --model_name_or_path bert-base-uncased \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir /result/SimSCE12_bert \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_train \
    --fp16 \
```

#### Evaluate the RobustEmbed embeddings on STS and Transfer tasks
```bash
python -m torch.distributed.launch --nproc_per_node 4 --master_port $(expr $RANDOM + 1000) train2.py \
    --model_name_or_path /result/SimSCE12_bert \
    --train_file data/wiki1m_for_simcse.txt \
    --output_dir /result/SimSCE12_bert \
    --num_train_epochs 1 \
    --per_device_train_batch_size 64 \
    --learning_rate 3e-5 \
    --max_seq_length 32 \
    --evaluation_strategy steps \
    --metric_for_best_model eval_avg_sts \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls \
    --mlp_only_train \
    --overwrite_output_dir \
    --temp 0.05 \
    --do_eval \
    --fp16 \
```

#### Evaluate the RobustEmbed embeddings on STS and Transfer tasks

```python
import textattack
import random
import transformers
import datasets
from adversarial_fine_tunning import BertForAT
from datasets import load_dataset


mnli_dataset = load_dataset('imdb') #load different dataset
train_dataset = textattack.datasets.HuggingFaceDataset(mnli_dataset['train'].shuffle())
eval_dataset = textattack.datasets.HuggingFaceDataset(mnli_dataset['test'].shuffle())


model_name = '/result/SimSCE12_bert'
config = transformers.AutoConfig.from_pretrained(pretrained_model_name_or_path = model_name, num_labels=num_labels)
model = BertForAT.from_pretrained(pretrained_model_name_or_path = model_name, config=config)         
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, do_lower_case= True)
model_wrapper = textattack.models.wrappers.HuggingFaceModelWrapper(model, tokenizer)

training_args = textattack.TrainingArgs(
    num_epochs=3,
    parallel=True,
    learning_rate=5e-5, #1e-5
    per_device_train_batch_size=32,
    gradient_accumulation_steps=8,
    log_to_tb=True,
)

trainer = textattack.Trainer(
    model_wrapper,
    "classification", # regression, classification
    None,
    train_dataset,
    eval_dataset,
    training_args
)
trainer.train()


#attack = textattack.attack_recipes.PWWSRen2019.build(trainer.model_wrapper)
attack = textattack.attack_recipes.TextFoolerJin2019.build(trainer.model_wrapper)
#attack = textattack.attack_recipes.TextBuggerLi2018.build(trainer.model_wrapper)
#attack = textattack.attack_recipes.BAEGarg2019.build(trainer.model_wrapper)
#attack = textattack.attack_recipes.BERTAttackLi2020.build(trainer.model_wrapper)

attack_args = textattack.AttackArgs(num_examples=1000, disable_stdout=True)
attacker = textattack.Attacker(attack, eval_dataset, attack_args)
attacker.attack_dataset()
```
