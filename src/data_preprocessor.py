"""
Data preprocessing for transformer-based recommendation system
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder
import pickle
import os

class RecommendationDataPreprocessor:
    """Preprocess Amazon Beauty dataset for transformer-based recommendations"""
    
    def __init__(self, min_interactions: int = 5, max_sequence_length: int = 50):
        self.min_interactions = min_interactions
        self.max_sequence_length = max_sequence_length
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.user_to_id = {}
        self.item_to_id = {}
        self.id_to_user = {}
        self.id_to_item = {}
        
    def filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter users and items with minimum interactions"""
        print("Filtering data...")
        
        # Filter items with minimum interactions
        item_counts = df['asin'].value_counts()
        valid_items = item_counts[item_counts >= self.min_interactions].index
        df = df[df['asin'].isin(valid_items)]
        
        # Filter users with minimum interactions
        user_counts = df['reviewerID'].value_counts()
        valid_users = user_counts[user_counts >= self.min_interactions].index
        df = df[df['reviewerID'].isin(valid_users)]
        
        print(f"After filtering: {len(df):,} interactions, "
              f"{df['reviewerID'].nunique():,} users, "
              f"{df['asin'].nunique():,} items")
        
        return df
    
    def encode_ids(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode user and item IDs to consecutive integers"""
        print("Encoding user and item IDs...")
        
        # Encode users and items
        df['user_id'] = self.user_encoder.fit_transform(df['reviewerID'])
        df['item_id'] = self.item_encoder.fit_transform(df['asin'])
        
        # Create mapping dictionaries
        self.user_to_id = dict(zip(df['reviewerID'], df['user_id']))
        self.item_to_id = dict(zip(df['asin'], df['item_id']))
        self.id_to_user = {v: k for k, v in self.user_to_id.items()}
        self.id_to_item = {v: k for k, v in self.item_to_id.items()}
        
        return df
    
    def create_sequences(self, df: pd.DataFrame) -> Dict[str, List]:
        """Create user interaction sequences sorted by timestamp"""
        print("Creating user sequences...")
        
        # Sort by user and timestamp
        df = df.sort_values(['user_id', 'unixReviewTime'])
        
        sequences = []
        labels = []
        user_ids = []
        
        for user_id in df['user_id'].unique():
            user_data = df[df['user_id'] == user_id]
            items = user_data['item_id'].tolist()
            
            # Create sequences of different lengths for training
            for i in range(1, len(items)):
                if i <= self.max_sequence_length:
                    sequence = items[:i]
                    target = items[i]
                    
                    # Pad sequence if necessary
                    if len(sequence) < self.max_sequence_length:
                        sequence = [0] * (self.max_sequence_length - len(sequence)) + sequence
                    
                    sequences.append(sequence)
                    labels.append(target)
                    user_ids.append(user_id)
        
        return {
            'sequences': sequences,
            'labels': labels,
            'user_ids': user_ids
        }
    
    def train_test_split(self, data: Dict[str, List], test_ratio: float = 0.2) -> Tuple[Dict, Dict]:
        """Split data into train and test sets"""
        print("Splitting data into train/test sets...")
        
        n_samples = len(data['sequences'])
        n_test = int(n_samples * test_ratio)
        
        # Random split
        indices = np.random.permutation(n_samples)
        test_indices = indices[:n_test]
        train_indices = indices[n_test:]
        
        train_data = {
            'sequences': [data['sequences'][i] for i in train_indices],
            'labels': [data['labels'][i] for i in train_indices],
            'user_ids': [data['user_ids'][i] for i in train_indices]
        }
        
        test_data = {
            'sequences': [data['sequences'][i] for i in test_indices],
            'labels': [data['labels'][i] for i in test_indices],
            'user_ids': [data['user_ids'][i] for i in test_indices]
        }
        
        print(f"Train samples: {len(train_data['sequences']):,}")
        print(f"Test samples: {len(test_data['sequences']):,}")
        
        return train_data, test_data
    
    def save_preprocessed_data(self, train_data: Dict, test_data: Dict, 
                              output_dir: str = "data") -> None:
        """Save preprocessed data and encoders"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        with open(os.path.join(output_dir, "train_data.pkl"), "wb") as f:
            pickle.dump(train_data, f)
        
        with open(os.path.join(output_dir, "test_data.pkl"), "wb") as f:
            pickle.dump(test_data, f)
        
        # Save encoders and mappings
        encoders = {
            'user_encoder': self.user_encoder,
            'item_encoder': self.item_encoder,
            'user_to_id': self.user_to_id,
            'item_to_id': self.item_to_id,
            'id_to_user': self.id_to_user,
            'id_to_item': self.id_to_item,
            'n_users': len(self.user_to_id),
            'n_items': len(self.item_to_id),
            'max_sequence_length': self.max_sequence_length
        }
        
        with open(os.path.join(output_dir, "encoders.pkl"), "wb") as f:
            pickle.dump(encoders, f)
        
        print(f"Preprocessed data saved to {output_dir}/")
    
    def process_dataset(self, reviews_df: pd.DataFrame, output_dir: str = "data") -> Tuple[Dict, Dict]:
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Filter data
        filtered_df = self.filter_data(reviews_df)
        
        # Encode IDs
        encoded_df = self.encode_ids(filtered_df)
        
        # Create sequences
        sequence_data = self.create_sequences(encoded_df)
        
        # Split data
        train_data, test_data = self.train_test_split(sequence_data)
        
        # Save data
        self.save_preprocessed_data(train_data, test_data, output_dir)
        
        print("Data preprocessing completed!")
        return train_data, test_data

if __name__ == "__main__":
    # Load raw data
    reviews_df = pd.read_csv("data/reviews.csv")
    
    # Preprocess data
    preprocessor = RecommendationDataPreprocessor(min_interactions=5, max_sequence_length=20)
    train_data, test_data = preprocessor.process_dataset(reviews_df)
    
    print("\n=== Preprocessing Summary ===")
    print(f"Number of users: {preprocessor.user_encoder.classes_.shape[0]:,}")
    print(f"Number of items: {preprocessor.item_encoder.classes_.shape[0]:,}")
    print(f"Training sequences: {len(train_data['sequences']):,}")
    print(f"Test sequences: {len(test_data['sequences']):,}")
    print(f"Max sequence length: {preprocessor.max_sequence_length}")
