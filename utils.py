import torch
from datasets import load_dataset,load_metric
from opacus.privacy_engine import PrivacyEngine
from torch import nn
from tqdm import tqdm
from private_transformers import PrivacyEngine as PrivacyEngine_2
import torch.nn.functional as F
from transformers import BertTokenizer
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda:0")
tokenizer = BertTokenizer.from_pretrained("prajjwal1/bert-tiny")

def collate_fn(batch):
    # print(batch)
    input_ids = [item["input_ids"].clone().detach() for item in batch]
    attention_masks = [item["attention_mask"].clone().detach() for item in batch]
    token_type_ids = [item["token_type_ids"].clone().detach() for item in batch]
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "token_type_ids": token_type_ids,
        "labels": labels
    }

def preprocess_data_sst2(data):
    return tokenizer(data["sentence"], truncation=True, padding="max_length",max_length=128)

def preprocess_data_qnli(data):
    return tokenizer(data["question"], data["sentence"], padding="max_length", truncation="only_second", max_length=128)

def preprocess_data_qqp(data):
    return tokenizer(data["question1"], data["question2"], padding="max_length", truncation="only_second", max_length=128)

def preprocess_data_mnli(data):
    return tokenizer(data["premise"], data["hypothesis"], padding="max_length", truncation="longest_first", max_length=128)

def load_cleaned_data(dataset_name):
    dataset = load_dataset("glue", dataset_name)
    # SST-2 dataset
    if dataset_name == 'sst2':
        tokenized_data = dataset.map(preprocess_data_sst2, batched=True)
        tokenized_data = tokenized_data.remove_columns(["idx","sentence"])
        tokenized_data = tokenized_data.rename_column("label", "labels")
    if dataset_name == 'qnli':
        tokenized_data = dataset.map(preprocess_data_qnli, batched=True)
        tokenized_data = tokenized_data.remove_columns(["idx","sentence","question"])
        tokenized_data = tokenized_data.rename_column("label", "labels")
    if dataset_name == 'qqp':
        tokenized_data = dataset.map(preprocess_data_qqp, batched=True)
        tokenized_data = tokenized_data.remove_columns(["idx","question1","question2"])
        tokenized_data = tokenized_data.rename_column("label", "labels")
    if dataset_name == 'mnli':
        tokenized_data = dataset.map(preprocess_data_mnli, batched=True)
        tokenized_data = tokenized_data.remove_columns(["idx","premise","hypothesis",])
        tokenized_data = tokenized_data.rename_column("label", "labels")

    tokenized_data.set_format("torch")
    return tokenized_data

def count_trainable_params(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_params

def evaluate_model(model,dataloader,task):
    metric = load_metric("glue", task) 
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    return metric.compute()

def evaluate_model_2(model,dataloader,task):
    metric = load_metric("glue", task) 
    model.eval()
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(batch['input_ids'],batch['attention_mask'])
        logits = outputs
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])
    return metric.compute()

def trainModel(model,optimizer,train_dataloader,val_dataloader,loss_fn,lr_scheduler,tqdm,task,epochs=5):

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step,batch in enumerate(tqdm(train_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)

            loss = loss_fn(outputs.logits, batch["labels"])
            total_loss += loss.detach().float()

            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} ")

        # Evaluate on validation set
        with torch.no_grad():
            val_accuracy = evaluate_model(model, val_dataloader,task)
            print(f"Epoch {epoch+1}, Validation Accuracy without DP: {val_accuracy}")

    print("Training complete!")

def trainModel_2(model,optimizer,train_dataloader,val_dataloader,loss_fn,lr_scheduler,tqdm,task,epochs=5):

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step,batch in enumerate(tqdm(train_dataloader)):

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'],batch['attention_mask'])
            loss = loss_fn(outputs, batch["labels"])
            total_loss += loss.detach().float()

            loss.backward()
            optimizer.step()

            lr_scheduler.step()
            optimizer.zero_grad()

        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} ")

        # Evaluate on validation set
        with torch.no_grad():
            val_accuracy = evaluate_model_2(model, val_dataloader,task)
            print(f"Epoch {epoch+1}, Validation Accuracy without DP: {val_accuracy}")
    print("Training complete!")


def dp_train(model,train_dataloader,tokenized_data,optimizer,lr_scheduler,epoch_num,val_dataloader,dataset_name):
    privacy_engine = PrivacyEngine()
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    dp_model,optimizer,dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_epsilon = 8,
        target_delta = 1/tokenized_data['train'].num_rows,
        epochs = 5,
        max_grad_norm=0.2,
    )

    # for i, param in enumerate(dp_model.params):
    #     print(f"Layer {i}: grad_sample shape = {param.grad_sample.shape if param.grad_sample is not None else 'None'}")
    for epoch in range(epoch_num):
        dp_model.train()
        for step,batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            # Forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            # input_ids = pad_sequence(batch['input_ids'], batch_first=True, padding_value=0)
            # token_type_ids = pad_sequence(batch['token_type_ids'], batch_first=True, padding_value=0)
            # attention_mask = pad_sequence(batch['attention_mask'], batch_first=True, padding_value=0)

            outputs = dp_model(**batch)
            # for name, module in dp_model.named_modules():
            #     def hook(module, input, output, name=name):  # 捕获name变量的当前值
            #         # print(f"Hook Triggered for {name}")
            #         print(f"Input shape: {input[0].shape if input else 'None'}, Output shape: {output.shape}")
            #     module.register_forward_hook(hook)
            # outputs = dp_model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = loss_fn(outputs.logits, batch["labels"])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
        # Evaluate on validation set
        with torch.no_grad():
            val_accuracy = evaluate_model(dp_model, val_dataloader,dataset_name)
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy}")

    print("Training complete")

def dp_train_2(model,train_dataloader,tokenized_data,optimizer,lr_scheduler,epoch_num,val_dataloader,dataset_name):
    privacy_engine = PrivacyEngine_2(
        model,
        batch_size=1024,
        sample_size=tokenized_data['train'].num_rows,
        epochs=5,
        max_grad_norm=0.2,
        target_epsilon=8,
        clipping_mode="ghost"
    )
    privacy_engine.attach(optimizer)
    for epoch in range(5):
            model.train()
            # total_loss = 0
            for step,batch in enumerate(tqdm(train_dataloader)):

                optimizer.zero_grad()
                # Forward pass
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

                loss = F.cross_entropy(outputs.logits, batch["labels"]).mean(dim=0).unsqueeze(0)
                # Backward pass and update with DP
                optimizer.step(loss=loss)

                lr_scheduler.step()


            # Evaluate on validation set
            with torch.no_grad():
                val_accuracy = evaluate_model(model, val_dataloader,dataset_name)
                print(f"Epoch {epoch+1}, Validation Accuracy DP: {val_accuracy}")
    print("Training complete")

def dp_train_3(model,train_dataloader,tokenized_data,optimizer,lr_scheduler,epoch_num,val_dataloader,dataset_name):
    privacy_engine = PrivacyEngine()
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    lora_dp_model,optimizer,dataloader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=train_dataloader,
        target_epsilon = 8,
        target_delta = 1/tokenized_data['train'].num_rows,
        epochs = 5,
        max_grad_norm=0.2,
    )

    for epoch in range(epoch_num):
        lora_dp_model.train()
        for step,batch in enumerate(tqdm(dataloader)):
            optimizer.zero_grad()
            # Forward pass
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch['input_ids'],batch['attention_mask'])
            loss = loss_fn(outputs, batch["labels"])
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
        # Evaluate on validation set
        with torch.no_grad():
            val_accuracy = evaluate_model_2(lora_dp_model, val_dataloader,dataset_name)
            print(f"Epoch {epoch+1}, Validation Accuracy: {val_accuracy}")

    print("Training complete")


