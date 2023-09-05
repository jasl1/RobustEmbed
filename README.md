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
import markdown
html = markdown.markdown(your_text_string)
```
