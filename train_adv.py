import numpy as np
import torch
import os
import argparse
import json
from datetime import datetime
import random
import logging
import os
import datasets
from tqdm import tqdm
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
from torch.utils.data import DataLoader
from data_collator import DataCollatorCustom
from transformers import default_data_collator, get_linear_schedule_with_warmup

start_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

def main(args):
    device = args.device
    batch_size = args.batch_size

    # data preprocessing
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    special_tokens_dict = {'sep_token': '[SEP]'}
    tokenizer.add_special_tokens(special_tokens_dict)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_dataset = DataCollatorCustom(filename=args.train_data_path,
                                       tokenizer=tokenizer,
                                       max_length=args.max_length)
    train_dataset = datasets.Dataset.from_dict(train_dataset.load_dataset())
    print("Input:", train_dataset["input_ids"][0])
    print("Label:", train_dataset["labels"][0])
    print("Length:", len(train_dataset["input_ids"][0]), \
        len(train_dataset["labels"][0]))

    eval_dataset = DataCollatorCustom(filename=args.valid_data_path,
                                      tokenizer=tokenizer,
                                      max_length=args.max_length)
    eval_dataset = datasets.Dataset.from_dict(eval_dataset.load_dataset())

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )

    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=batch_size, pin_memory=True
        )

    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
    
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    
    model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(model, peft_config)

    # optimizer and lr scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    logging.info("==========Configuration==========")
    logging.info(peft_config)

    model.print_trainable_parameters()
    model = model.to(device)

    # training
    logging.info("==========Start training==========")

    peft_model_id = f"{args.model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}"

    comparative_loss = -1

    for epoch in tqdm(range(args.num_epochs), position=0, desc="Epoch", leave=False):
        model.train()
        total_loss = 0

        for batch in tqdm(train_dataloader, position=1, desc="Training", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            
            inputs = {"attention_mask": batch["attention_mask"], "labels": batch["labels"]}
            
            embeds_init = model.transformer.wte(batch["input_ids"])
            
            if args.adv_init_mag > 0:
                input_mask = inputs['attention_mask'].to(embeds_init)
                input_lengths = torch.sum(input_mask, 1)

                if args.norm_type == "l2":
                    delta = torch.zeros_like(embeds_init).normal_(0, 1) * input_mask.unsqueeze(2)
                    dims = input_lengths * embeds_init.size(-1)
                    mag = args.adv_init_mag / torch.sqrt(dims)
                    delta = (delta * mag.view(-1, 1, 1)).detach()
            else:
                delta = torch.zeros_like(embeds_init)
            
            outputs = model(**batch)
            
            # K ascent steps
            for astep in range(args.adv_steps):
                # [1] forward pass
                delta.requires_grad_()
                inputs['inputs_embeds'] = delta + embeds_init
                
                adv_outputs = model(**inputs)
                adv_loss = adv_outputs.loss
                mse_loss = F.mse_loss(adv_outputs.logits, outputs.logits.detach(), reduction="mean")
                adv_loss = adv_loss + args.div_step_size * mse_loss
                adv_loss = adv_loss / args.adv_steps
                total_loss += adv_loss.detach().float()
                
                # [2] backward pass
                adv_loss.backward(retain_graph=True)

                if astep == args.adv_steps - 1:
                    break
                
                # [3] get gradient on delta of divergence loss
                delta_grad_2, = torch.autograd.grad(mse_loss, delta)
                
                # [4] get full gradients of delta
                full_delta_grad = delta_grad_2

                # [5] update and clip
                if args.norm_type == "l2":
                    denorm = torch.norm(full_delta_grad.view(full_delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
                    denorm = torch.clamp(denorm, min=1e-8)
                    delta = (delta + args.adv_lr * full_delta_grad / denorm).detach()
                    if args.adv_max_norm > 0:
                        delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).detach()
                        exceed_mask = (delta_norm > args.adv_max_norm).to(embeds_init)
                        reweights = (args.adv_max_norm / delta_norm * exceed_mask \
                                     + (1 - exceed_mask)).view(-1, 1, 1)
                        delta = (delta * reweights).detach()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        print("train_loss:", total_loss, len(train_dataloader), total_loss / len(train_dataloader))

        model.eval()
        eval_loss = 0
        eval_preds = []
        for batch in tqdm(eval_dataloader, position=1, desc="Validation", leave=False):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(), skip_special_tokens=True)
            )
        
        print("eval_loss:", eval_loss, len(eval_dataloader), eval_loss / len(eval_dataloader))
        
        eval_epoch_loss = eval_loss / len(eval_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)

        logging.info(f"Epoch {epoch+1}: \
                    train_loss: {train_epoch_loss}, \
                    train_ppl: {train_ppl}, \
                    valid_loss: {eval_epoch_loss}, \
                    valid_ppl: {eval_ppl}"
                )

        cummulative_loss = eval_loss / len(eval_dataloader)

        if comparative_loss == -1 or cummulative_loss < comparative_loss:
            # Update comparative_loss for later comparison
            comparative_loss = cummulative_loss
            
            # Saving model
            logging.info(f"Epoch {epoch+1}: Saving model and tokenizer...")
            model.save_pretrained(os.path.join("new", peft_model_id))
            tokenizer.save_pretrained(os.path.join("new", peft_model_id))
            logging.info(f"Epoch {epoch+1}: Done.")
        
        # Generate new train dataset after each epoch to diversify the training set
        # data_preparation(args)
        # train_dataset = DataCollatorCustom(filename=args.train_data_path,
        #                                tokenizer=tokenizer,
        #                                max_length=args.max_length)
        # train_dataset = datasets.Dataset.from_dict(train_dataset.load_dataset())
        # train_dataloader = DataLoader(
        #     train_dataset,
        #     shuffle=True,
        #     collate_fn=default_data_collator,
        #     batch_size=batch_size,
        #     pin_memory=True,
        #     )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str,
                        default='./data/quora/QQP_split/train_ready.txt',
                        help='train dataset file')
    parser.add_argument('--valid_data_path', type=str,
                        default='./data/quora/QQP_split/valid_ready.txt',
                        help='valid dataset file')

    parser.add_argument('--input_file', type=str,
                        default='./data/quora/QQP_split/train.txt',
                        help='input dataset file')
    parser.add_argument('--output_file', type=str,
                        default='./data/quora/QQP_split/train_ready.txt',
                        help='output dataset file')

    parser.add_argument('--log', type=str,
                        default='./logs/train_{datetime}.log',
                        help='Log filename')
    parser.add_argument('--device', type=str, default='cuda',
                        help='{cuda, cpu}')

    parser.add_argument('--model_name_or_path', type=str, default="ai-forever/mGPT",
                        help='pretrained model name')
    parser.add_argument('--max_length', type=int, default=64,
                        help='Maximum number of tokens for each sequence')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                        help='Evaluation batch size')

    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate of fine-tuning')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--train_size', type=int, default=200000)
    
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    
    # ========================= ADVERSARIAL CONFIGURATION =========================
    parser.add_argument('--adv_lr', type=float, default=2e-3)
    parser.add_argument('--adv_steps', type=int, default=2, help="should be at least 1")
    parser.add_argument('--adv_init_mag', type=float, default=1)
    parser.add_argument('--norm_type', type=str, default="l2")
    parser.add_argument('--adv_max_norm', type=float, default=2e-5, help="set to 0 to be unlimited")
    parser.add_argument('--div_step_size', type=float, default=10)
    # ===========================================================================

    parser.add_argument('--debug', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=2023)
    parser.add_argument('--type', type=str, default="train")

    args = parser.parse_args()

    if os.path.exists("logs/") == False:
        os.mkdir("logs/")

    log_format = '%(asctime)s [%(levelname)s] %(message)s'
    log_level = logging.DEBUG if args.debug else logging.INFO
    log_file = args.log

    logging.basicConfig(level=log_level, format=log_format,
                        filename=log_file.format(datetime=start_datetime.replace(':','-')))
    logging.getLogger().setLevel(log_level)

    # Reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    logging.info(f'Parsed args: {json.dumps(dict(args.__dict__), indent=2)}')

    main(args)