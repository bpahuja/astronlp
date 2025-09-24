import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence
import numpy as np
import pandas as pd
from datetime import datetime
from torch.amp import autocast, GradScaler
import json
import pickle
from concurrent.futures import ThreadPoolExecutor


class ParagraphDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

    def chunk_text(self, text):
        # Use the tokenizer's encode method for proper handling
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        
        for i in range(0, len(tokens), self.max_length - 2):
            chunk_tokens = tokens[i:i + self.max_length - 2]
            # Add CLS and SEP tokens
            chunk_with_special = [self.tokenizer.cls_token_id] + chunk_tokens + [self.tokenizer.sep_token_id]
            chunks.append(torch.tensor(chunk_with_special, dtype=torch.long))
        
        # Ensure at least one chunk exists
        if not chunks:
            chunks = [torch.tensor([self.tokenizer.cls_token_id, self.tokenizer.sep_token_id], dtype=torch.long)]
            
        return chunks

    def __getitem__(self, idx):
        chunks = self.chunk_text(self.texts[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return chunks, label

    def __len__(self):
        return len(self.texts)

class BERTMetaClassifier(nn.Module):
    def __init__(self, model_name, cache_path, hidden_dim=768, meta_hidden=256, num_classes=2, dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, cache_dir=cache_path)
        self.lstm = nn.LSTM(hidden_dim, meta_hidden, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(meta_hidden * 2, num_classes)
        self.pad_token_id = self.bert.config.pad_token_id
    
    def forward(self, input_ids_list, attention_mask_list, paragraph_sizes):
        # Validate inputs
        assert len(input_ids_list) == len(attention_mask_list), "Input IDs and attention masks must have same length"
        assert sum(paragraph_sizes) == len(input_ids_list), "Paragraph sizes must sum to total number of chunks"
        
        # Process BERT embeddings - no autocast for ONNX compatibility
        bert_output = self.bert(input_ids=input_ids_list, attention_mask=attention_mask_list)
        cls_embeddings = bert_output.last_hidden_state[:, 0, :]  # [num_chunks, hidden_dim]
        
        # Create paragraph assignment tensor vectorized
        device = cls_embeddings.device
        paragraph_sizes_tensor = torch.tensor(paragraph_sizes, device=device, dtype=torch.long)
        paragraph_indices = torch.repeat_interleave(
            torch.arange(len(paragraph_sizes_tensor), device=device), 
            paragraph_sizes_tensor
        )  # [num_chunks] - maps each chunk to its paragraph index
        
        # Process each paragraph through LSTM using vectorized operations
        max_para_size = max(paragraph_sizes)
        num_paragraphs = len(paragraph_sizes)
        
        # Create batched input for LSTM [num_paragraphs, max_seq_len, hidden_dim]
        lstm_input = torch.zeros(num_paragraphs, max_para_size, cls_embeddings.size(-1), 
                                device=device, dtype=cls_embeddings.dtype)
        
        # Fill the batched input using advanced indexing
        chunk_idx = 0
        for para_idx, para_size in enumerate(paragraph_sizes):
            lstm_input[para_idx, :para_size] = cls_embeddings[chunk_idx:chunk_idx + para_size]
            chunk_idx += para_size
        
        # Create attention mask for LSTM to ignore padding
        lstm_mask = torch.zeros(num_paragraphs, max_para_size, device=device, dtype=torch.bool)
        for para_idx, para_size in enumerate(paragraph_sizes):
            lstm_mask[para_idx, :para_size] = True
        
        # Run LSTM on batched input
        lstm_output, _ = self.lstm(lstm_input)  # [num_paragraphs, max_seq_len, hidden_dim*2]
        
        # Vectorized mean pooling with masking
        # Expand mask to match lstm_output dimensions
        lstm_mask_expanded = lstm_mask.unsqueeze(-1).expand_as(lstm_output)  # [num_paragraphs, max_seq_len, hidden_dim*2]
        
        # Zero out padded positions
        masked_lstm_output = lstm_output * lstm_mask_expanded.float()
        
        # Sum over sequence dimension and divide by actual lengths
        paragraph_sums = masked_lstm_output.sum(dim=1)  # [num_paragraphs, hidden_dim*2]
        paragraph_lengths = paragraph_sizes_tensor.float().unsqueeze(-1)  # [num_paragraphs, 1]
        paragraph_embeddings = paragraph_sums / paragraph_lengths  # [num_paragraphs, hidden_dim*2]
        
        # Apply dropout and classify
        paragraph_embeddings = self.dropout(paragraph_embeddings)
        return self.classifier(paragraph_embeddings)


def single_sample_collate_fn(batch):
    """For batch_size=1, return the single sample directly"""
    return batch[0]


def collate_fn(batch):
    # Flatten all chunks across all paragraphs
    all_chunks = []
    all_attn = []
    paragraph_sizes = []
    labels = []

    for chunks, label in batch:
        all_chunks.extend(chunks)
        all_attn.extend([torch.ones_like(chunk) for chunk in chunks])
        paragraph_sizes.append(len(chunks))
        labels.append(label)

    # Pad to [total_chunks, seq_len]
    input_ids_batch = pad_sequence(all_chunks, batch_first=True, padding_value=0)
    attention_mask_batch = pad_sequence(all_attn, batch_first=True, padding_value=0)
    labels = torch.stack(labels)

    return input_ids_batch, attention_mask_batch, paragraph_sizes, labels


class BERTMetaClassifierTrainer:
    def __init__(self, 
                 model_name='allenai/scibert_scivocab_uncased',
                 cache_path='/vol/bitbucket/bp824/hf_models/scibert',
                 output_dir='./output',
                 config=None):
        """
        Initialize the trainer with fixed hyperparameters
        
        Args:
            model_name: Pretrained model name
            cache_path: Path to cache pretrained models
            output_dir: Directory to save outputs
            config: Dictionary with training configuration
        """
        self.model_name = model_name
        self.cache_path = cache_path
        self.output_dir = output_dir
        
        # Default configuration
        self.config = {
            'learning_rate': 1e-5,
            'batch_size': 4,
            'meta_hidden': 256,
            'dropout': 0.2,
            'epochs': 5,
            'optimizer': 'adam',
            'max_length': 512,
            'num_classes': 2,
            'test_size': 0.2,
            'random_state': 42
        }
        
        # Update with provided config
        if config:
            self.config.update(config)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        torch.backends.cudnn.benchmark = True
        
        # Initialize components
        self.model = None
        self.tokenizer = None
        self.optimizer = None
        self.scaler = GradScaler()
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Generate timestamp for this training run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"Trainer initialized with config: {self.config}")
        print(f"Using device: {self.device}")

    def load_model_and_tokenizer(self):
        """Load the model and tokenizer"""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, 
            cache_dir=self.cache_path, 
            use_fast=True
        )
        
        self.model = BERTMetaClassifier(
            model_name=self.model_name,
            cache_path=self.cache_path,
            hidden_dim=768,
            meta_hidden=self.config['meta_hidden'],
            dropout=self.config['dropout'],
            num_classes=self.config['num_classes']
        )
        
        self.model.to(self.device)
        
        # Setup optimizer
        if self.config['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config['learning_rate'])
        elif self.config['optimizer'] == 'sgd':
            self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.config['learning_rate'], momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config['optimizer']}")
        
        print("Model and tokenizer loaded successfully!")

    def preprocess_data(self, data_path):
        """Load and preprocess the data"""
        print(f"Loading data from: {data_path}")
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"File not found: {data_path}")
        
        # Load JSONL
        df = pd.read_json(data_path, lines=True)
        df["label"] = df["label"].map({"methodology": 1, "not_methodology": 0})
        
        # Define preprocessing function
        def preprocess_row(row):
            paragraph = row["paragraph"]
            clean_paragraph = paragraph.replace("[MATH_tex=", "[MATH]").split("]")[-1]
            return f"{row['section']} - {clean_paragraph}"
        
        # Balance the dataset
        methodology_df = df[df['label'] == 1]
        non_methodology_df = df[df['label'] == 0]
        
        min_count = min(len(methodology_df), len(non_methodology_df))
        methodology_sampled = methodology_df.sample(n=min_count, random_state=self.config['random_state'])
        non_methodology_sampled = non_methodology_df.sample(n=min_count, random_state=self.config['random_state'])
        
        balanced_df = pd.concat([methodology_sampled, non_methodology_sampled]).sample(
            frac=1, random_state=self.config['random_state']
        ).reset_index(drop=True)
        
        # Apply preprocessing in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            balanced_df["input_text"] = list(executor.map(preprocess_row, balanced_df.to_dict("records")))
        
        # Split the data
        train_df, test_df = train_test_split(
            balanced_df, 
            test_size=self.config['test_size'], 
            stratify=balanced_df["label"], 
            random_state=self.config['random_state']
        )
        
        print(f"Data loaded: {len(train_df)} train samples, {len(test_df)} test samples")
        print(f"Class distribution - Train: {train_df['label'].value_counts().to_dict()}")
        print(f"Class distribution - Test: {test_df['label'].value_counts().to_dict()}")
        
        return train_df, test_df

    def create_dataloaders(self, train_df, test_df):
        """Create data loaders"""
        print("Creating datasets and dataloaders...")
        
        train_dataset = ParagraphDataset(
            train_df["input_text"].tolist(), 
            train_df["label"].tolist(), 
            self.tokenizer,
            max_length=self.config['max_length']
        )
        
        test_dataset = ParagraphDataset(
            test_df["input_text"].tolist(), 
            test_df["label"].tolist(), 
            self.tokenizer,
            max_length=self.config['max_length']
        )
        
        # Use appropriate collate function based on batch size
        if self.config['batch_size'] == 1:
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=1, 
                shuffle=True, 
                collate_fn=single_sample_collate_fn, 
                pin_memory=True
            )
            test_dataloader = DataLoader(
                test_dataset, 
                batch_size=1, 
                shuffle=False, 
                collate_fn=single_sample_collate_fn
            )
        else:
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size=self.config['batch_size'], 
                shuffle=True, 
                collate_fn=collate_fn, 
                pin_memory=True
            )
            test_dataloader = DataLoader(
                test_dataset, 
                batch_size=1, 
                shuffle=False, 
                collate_fn=single_sample_collate_fn
            )
        
        return train_dataloader, test_dataloader

    def train_epoch(self, dataloader):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in dataloader:
            if len(batch) == 2:  # Single sample batch
                chunks, label = batch
                input_ids_list = chunks
                attention_mask_list = [torch.ones_like(chunk, dtype=torch.long) for chunk in chunks]
                paragraph_sizes = [len(chunks)]
                labels = label.unsqueeze(0)
            else:  # Multi-sample batch
                input_ids_list, attention_mask_list, paragraph_sizes, labels = batch
                
            input_ids_list = [x.to(self.device) for x in input_ids_list]
            attention_mask_list = [x.to(self.device) for x in attention_mask_list]
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = self.model(input_ids_list, attention_mask_list, paragraph_sizes)
                loss = F.cross_entropy(outputs, labels)
            
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
            
        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in dataloader:
                if len(batch) == 2:  # Single sample batch
                    chunks, label = batch
                    input_ids_list = chunks
                    attention_mask_list = [torch.ones_like(chunk) for chunk in chunks]
                    paragraph_sizes = [len(chunks)]
                    labels = label.unsqueeze(0)
                else:  # Multi-sample batch
                    input_ids_list, attention_mask_list, paragraph_sizes, labels = batch
                    
                input_ids_list = [x.to(self.device) for x in input_ids_list]
                attention_mask_list = [x.to(self.device) for x in attention_mask_list]
                labels = labels.to(self.device)

                outputs = self.model(input_ids_list, attention_mask_list, paragraph_sizes)
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='macro')
        cm = confusion_matrix(all_labels, all_preds)
        
        return acc, f1, all_labels, all_preds, cm

    def train(self, data_path):
        """Main training loop"""
        print("Starting training...")
        
        # Load model and data
        self.load_model_and_tokenizer()
        train_df, test_df = self.preprocess_data(data_path)
        train_dataloader, test_dataloader = self.create_dataloaders(train_df, test_df)
        
        # Setup logging
        results_file = os.path.join(self.output_dir, f"training_results_{self.timestamp}.txt")
        training_history = []
        
        with open(results_file, 'w') as f:
            f.write(f"Training started at: {datetime.now()}\n")
            f.write(f"Configuration: {json.dumps(self.config, indent=2)}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Device: {self.device}\n")
            f.write("="*50 + "\n\n")
        
        best_f1 = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(self.config['epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['epochs']}")
            
            # Train
            train_loss = self.train_epoch(train_dataloader)
            
            # Evaluate
            train_acc, train_f1, _, _, _ = self.evaluate(train_dataloader)
            val_acc, val_f1, val_labels, val_preds, val_cm = self.evaluate(test_dataloader)
            
            # Log results
            epoch_results = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'train_f1': train_f1,
                'val_accuracy': val_acc,
                'val_f1': val_f1
            }
            training_history.append(epoch_results)
            
            log_msg = (f"Epoch {epoch + 1} | Train Loss: {train_loss:.4f} | "
                      f"Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f} | "
                      f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
            print(log_msg)
            
            with open(results_file, 'a') as f:
                f.write(log_msg + '\n')
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_model_state = self.model.state_dict().copy()
        
        # Final evaluation and save
        final_results = {
            'final_val_accuracy': val_acc,
            'final_val_f1': val_f1,
            'confusion_matrix': val_cm.tolist(),
            'training_history': training_history
        }
        
        # Save model and results
        model_path = self.save_model(best_model_state, final_results)
        
        print(f"\nTraining completed!")
        print(f"Best validation F1: {best_f1:.4f}")
        print(f"Final validation accuracy: {val_acc:.4f}")
        print(f"Final validation F1: {val_f1:.4f}")
        print(f"Model saved to: {model_path}")
        print(f"Results saved to: {results_file}")
        
        return model_path, final_results

    def save_model(self, model_state_dict, results):
        """Save the trained model and results"""
        model_dir = os.path.join(self.output_dir, f"model_{self.timestamp}")
        os.makedirs(model_dir, exist_ok=True)
        
        # Save model state dict
        model_path = os.path.join(model_dir, "model.pt")
        torch.save(model_state_dict, model_path)
        
        # Save tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        # Save configuration and results
        config_path = os.path.join(model_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                'model_config': self.config,
                'model_name': self.model_name,
                'cache_path': self.cache_path,
                'timestamp': self.timestamp
            }, f, indent=2)
        
        results_path = os.path.join(model_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save complete model info for easy loading
        model_info = {
            'model_path': model_path,
            'tokenizer_path': tokenizer_path,
            'config_path': config_path,
            'results_path': results_path,
            'model_config': self.config,
            'model_name': self.model_name
        }
        
        info_path = os.path.join(model_dir, "model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)
        
        return model_dir

    @classmethod
    def load_trained_model(cls, model_dir):
        """Load a previously trained model"""
        info_path = os.path.join(model_dir, "model_info.json")
        
        if not os.path.exists(info_path):
            raise FileNotFoundError(f"Model info file not found: {info_path}")
        
        with open(info_path, 'r') as f:
            model_info = json.load(f)
        
        # Create trainer instance
        trainer = cls(
            model_name=model_info['model_name'],
            cache_path=model_info.get('cache_path', ''),
            config=model_info['model_config']
        )
        
        # Load tokenizer
        trainer.tokenizer = AutoTokenizer.from_pretrained(model_info['tokenizer_path'])
        
        # Load model
        trainer.model = BERTMetaClassifier(
            model_name=model_info['model_name'],
            cache_path=model_info.get('cache_path', ''),
            hidden_dim=768,
            meta_hidden=model_info['model_config']['meta_hidden'],
            dropout=model_info['model_config']['dropout'],
            num_classes=model_info['model_config']['num_classes']
        )
        
        # Load model weights
        model_state = torch.load(model_info['model_path'], map_location=trainer.device)
        trainer.model.load_state_dict(model_state)
        trainer.model.to(trainer.device)
        trainer.model.eval()
        if model_info['model_config']['optimizer'] == 'adam':
            trainer.optimizer = torch.optim.Adam(trainer.model.parameters(), lr=model_info['model_config']['learning_rate'])
        elif model_info['model_config']['optimizer'] == 'adamw':
            trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=model_info['model_config']['learning_rate'])
        elif model_info['model_config']['optimizer'] == 'sgd':
            trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=model_info['model_config']['learning_rate'], momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {model_info['model_config']['optimizer']}")
        
        # Load results
        with open(model_info['results_path'], 'r') as f:
            results = json.load(f)
        
        print(f"Model loaded from: {model_dir}")
        print(f"Final validation F1: {results['final_val_f1']:.4f}")
        
        return trainer, results

def softmax_confidence(logits):
    probs = torch.softmax(logits, dim=-1)
    confidence, prediction = torch.max(probs, dim=-1)
    return confidence, prediction

def filter_by_confidence(logits, threshold):
    confidence, prediction = softmax_confidence(logits)
    mask = confidence > threshold
    return prediction[mask], mask

# Example usage
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='google-bert/bert-base-uncased')
    parser.add_argument('--cache_path', type=str, default="/vol/bitbucket/bp824/hf_models/bert-base")
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='/vol/bitbucket/bp824/data/bert-classifier-final/output')
    parser.add_argument('--bootstrap', type=bool, default=False, help='Enable bootstrapping for training')
    parser.add_argument('--unlabeled_data_path', type=str, default=None, help='Path to unlabeled data for bootstrapping')
    args = parser.parse_args()

    if args.bootstrap:
        print("Bootstrapping is enabled.")
        load_trainer, results = BERTMetaClassifierTrainer.load_trained_model(args.output_dir)


    
    # Custom configuration (optional)
    custom_config = {
        'learning_rate': 1e-5,
        'batch_size': 4,
        'meta_hidden': 256,
        'dropout': 0.2,
        'epochs': 3,
        'optimizer': 'adam'
    }
    
    # Initialize trainer
    trainer = BERTMetaClassifierTrainer(
        model_name=args.model_name,
        cache_path=args.cache_path,
        output_dir=args.output_dir,
        config=custom_config
    )
    
    # Train the model
    model_path, results = trainer.train(args.data_path)
    
    # Example of loading a trained model later
    # loaded_trainer, loaded_results = BERTMetaClassifierTrainer.load_trained_model(model_path)