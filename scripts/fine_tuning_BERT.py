import torch
#torch.cuda.empty_cache()
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from tqdm import trange
from tqdm.notebook import tqdm

import random


#############################################################

class BertFineTunningMultiClassText:
    def __init__(self,
                 X_train = None,
                 y_train = None,
                 pretrained_model:str = 'bert-base-uncased',
                 n_labels:int = 2,
                 val_size:float = 0.15,
                 batch_size:int = 8,
                 epochs = 5,
                 random_state:int = 42,
                 learning_rate:float = 1e-5,
                 eps:float = 1e-8,
                 warmup_steps:int = 0,
                 average:str = 'binary',
                 saving_path = None):
        
        

        self.data = X_train
        self.labels = y_train
        self.pretrained_model = pretrained_model
        self.n_labels = n_labels
        self.val_size = val_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.random_state = random_state
        self.learning_rate = learning_rate
        self.eps = eps
        self.warmup_steps = warmup_steps
        self.average = 'binary' if self.n_labels == 2 else average
        self.saving_path = saving_path

        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed(self.random_state)

        self.model, self.tokenizer, self.vector_size = self.get_model_tokenizer()
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                               lr = self.learning_rate,
                               eps = self.eps)
        
        if X_train != None:
            self.train_dataloader, self.val_dataloader = self.get_train_val_dataloaders()
        else:
            self.train_dataloader, self.val_dataloader = [], []

        self.model.cuda()

        self.scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                       num_warmup_steps = self.warmup_steps,
                                                       num_training_steps = len(self.train_dataloader)*self.epochs)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        
    
    def get_model_tokenizer(self):
        model = BertForSequenceClassification.from_pretrained(self.pretrained_model,
                                                              num_labels = self.n_labels,
                                                              output_attentions = False,
                                                              output_hidden_states = False)
        
        tokenizer = BertTokenizer.from_pretrained(self.pretrained_model,
                                                  do_lower_case = True)
        
        tensor_size = model.bert.embeddings.position_embeddings.num_embeddings

        return model, tokenizer, tensor_size

    def preprocess(self, input_text):
        return self.tokenizer.encode_plus(input_text,
                                          add_special_tokens = True,
                                          max_length = self.vector_size,
                                          truncation = True,
                                          padding = 'max_length',
                                          return_attention_mask = True,
                                          return_tensors = 'pt')
    
    def tokenize_text(self, texts, labels):
        token_id = []
        attention_masks = []

        for doc in texts:
            encoding_dict = self.preprocess(doc)
            token_id.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        token_id = torch.cat(token_id, dim = 0)
        attention_masks = torch.cat(attention_masks, dim = 0)
        
        if labels != None:
            labels = torch.tensor(labels)
        else:
            labels = None

        return token_id, attention_masks, labels
    
    def get_train_val_dataloaders(self):
        token_id, attention_mask, labels = self.tokenize_text(self.data, self.labels)

        train_idx, val_idx = train_test_split(np.arange(len(labels)),
                                              test_size= self.val_size,
                                              shuffle= True,
                                              stratify= labels,
                                              random_state= self.random_state)
        
        # train and validation sets

        train_set = TensorDataset(token_id[train_idx],
                                  attention_mask[train_idx],
                                  labels[train_idx])
        
        val_set = TensorDataset(token_id[val_idx],
                                attention_mask[val_idx],
                                labels[val_idx])
        
        # train and validation Dataloaders
        
        train_dataloader = DataLoader(train_set,
                                      sampler = RandomSampler(train_set),
                                      batch_size = self.batch_size)
        
        val_dataloader = DataLoader(val_set,
                                    sampler = SequentialSampler(val_set),
                                    batch_size = self.batch_size)
        
        return train_dataloader, val_dataloader
    
    def batch_metrics(self, pred_labels, true_labels):
        preds_flat = np.argmax(pred_labels, axis = 1).flatten()
        labels_flat = true_labels.flatten()

        b_accuracy = accuracy_score(labels_flat, preds_flat)
        b_precision = precision_score(labels_flat, preds_flat, average= self.average, zero_division= 0.)
        b_recall = recall_score(labels_flat, preds_flat, average= self.average)  
        b_f1 = f1_score(labels_flat, preds_flat, average= self.average) 

        return b_accuracy, b_precision, b_recall, b_f1

    def evaluation(self, dataloader):
        self.model.eval()

        predictions, true_vals = [], []

        loss_val_total = 0

        for batch in dataloader:
            batch = tuple(t.to(self.device) for t in batch)

            inputs = {
                'input_ids' : batch[0],
                'attention_mask' : batch[1],
                'labels' : batch[2]
            }

            with torch.no_grad():
                outputs = self.model(**inputs)

            loss = outputs[0]
            logits = outputs[1]
            loss_val_total += loss.item()

            logits = logits.detach().cpu().numpy()
            label_ids = inputs['labels'].cpu().numpy()
            predictions.append(logits)
            true_vals.append(label_ids)

        loss_val_avg = loss_val_total / len(dataloader)
        predictions = np.concatenate(predictions, axis= 0)
        true_vals = np.concatenate(true_vals, axis= 0)

        return loss_val_avg, predictions, true_vals
    
    def fine_tune(self):
        #torch.cuda.empty_cache()
        best_val_f1 = 0.0
        best_val_loss = 1.0

        for epoch in tqdm(range(1, self.epochs + 1)):
            self.model.train()

            train_loss = 0

            progress_bar = tqdm(self.train_dataloader,
                                desc = 'Epoch {:1d}'.format(epoch),
                                leave = False,
                                disable = False)
            
            for batch in progress_bar:
                self.model.zero_grad()

                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    'input_ids' : batch[0],
                    'attention_mask' : batch[1],
                    'labels' : batch[2]
                }

                train_outputs = self.model(**inputs)

                loss = train_outputs[0]
                train_loss += loss.item()
                loss.backward()
                self.optimizer.step()

                progress_bar.set_postfix({'training_loss' : '{:3f}'.format(loss.item() / len(batch))})

            tqdm.write(f'\nEpoch{epoch}')
            val_loss, val_preds, true_vals = self.evaluation(self.val_dataloader)
            val_accuracy, val_precision, val_recall, val_f1 = self.batch_metrics(val_preds, true_vals)

            tqdm.write(f'Validation loss: {val_loss}')
            tqdm.write(f'Accuracy: {val_accuracy}')
            tqdm.write(f'Precision ({self.average}): {val_precision}')
            tqdm.write(f'Recall ({self.average}): {val_recall}')
            tqdm.write(f'F1-score ({self.average}): {val_f1}')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), f'{self.saving_path}best_val_model.model')
                
                

        return
    
    def get_test_dataloader(self, X_test, y_test):
        token_id, attention_mask, labels = self.tokenize_text(X_test, y_test)

        
        
        # train and validation sets

        test_set = TensorDataset(token_id,
                                  attention_mask,
                                  labels)
        
        
        # train and validation Dataloaders
        
        test_dataloader = DataLoader(test_set,
                                    sampler = SequentialSampler(test_set),
                                    batch_size = self.batch_size)
        
        return test_dataloader


    def test(self, X_test, y_test = None, path_to_state_dict = None):
        torch.cuda.empty_cache()

        tuned_model = BertForSequenceClassification.from_pretrained(self.pretrained_model,
                                                                    num_labels = self.n_labels,
                                                                    output_attentions = False,
                                                                    output_hidden_states = False)
        tuned_model.to(self.device)

        if path_to_state_dict != None:
            tuned_model.load_state_dict(torch.load(path_to_state_dict + 'best_val_model.model',
                                        map_location = torch.device('cpu')))
        
        if y_test != None:
            test_dataloader = self.get_test_dataloader(X_test, y_test)
            _, test_preds, true_vals = self.evaluation(test_dataloader)
            test_accuracy, test_precision, test_recall, test_f1 = self.batch_metrics(test_preds, true_vals)

            tqdm.write(f'Accuracy: {test_accuracy}')
            tqdm.write(f'Precision ({self.average}): {test_precision}')
            tqdm.write(f'Recall ({self.average}): {test_recall}')
            tqdm.write(f'F1-score ({self.average}): {test_f1}')
            predictions = test_preds
        
        else:
            size = tuned_model.bert.embeddings.word_embeddings.embedding_dim

            for doc in X_test:
                if len(doc.split(' ')) > size:
                    doc = doc[:size]

                token_id, attention_mask, _ = self.tokenize_text([doc], y_test)

                with torch.no_grad():
                    output = tuned_model(token_id.to(self.device),
                                         token_type_ids = None,
                                         attention_mask = attention_mask.to(self.device))
                
                
                logits = output[1]
                prediction = logits.detach().cpu().numpy()

                predictions.append(prediction)
    
        return predictions

        