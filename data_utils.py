import torch
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from torch_geometric.data import HeteroData

class DataGenerator:
    def __init__(self, num_users=100, num_pois=500, num_categories=20):
        self.num_users = num_users
        self.num_pois = num_pois
        self.num_categories = num_categories
        
    def generate_sample_data(self):
        users = [f"user_{i}" for i in range(self.num_users)]
        pois = {}
        
        for i in range(self.num_pois):
            pois[f"poi_{i}"] = {
                "category": f"category_{np.random.randint(0, self.num_categories)}",
                "coords": (
                    40.7128 + np.random.uniform(-0.1, 0.1),
                    -74.0060 + np.random.uniform(-0.1, 0.1)
                )
            }
        
        check_ins = []
        for user in users:
            num_checkins = np.random.randint(10, 50)
            for _ in range(num_checkins):
                poi = np.random.choice(list(pois.keys()))
                timestamp = pd.Timestamp('2024-01-01') + pd.Timedelta(days=np.random.randint(0, 365))
                check_ins.append({
                    'user': user,
                    'poi': poi,
                    'category': pois[poi]['category'],
                    'coords': pois[poi]['coords'],
                    'timestamp': timestamp
                })
        
        return pd.DataFrame(check_ins)

class DataProcessor:
    def __init__(self, df):
        self.df = df
        self.users = sorted(df['user'].unique())
        self.pois = sorted(df['poi'].unique())
        self.categories = sorted(df['category'].unique())
        self.user_to_idx = {u: i for i, u in enumerate(self.users)}
        self.poi_to_idx = {p: i for i, p in enumerate(self.pois)}
        self.category_to_idx = {c: i for i, c in enumerate(self.categories)}
        
    def split_data(self, train_ratio=0.8):
        train_data = defaultdict(dict)
        test_data = defaultdict(dict)
        
        for user in self.users:
            user_data = self.df[self.df['user'] == user].sort_values('timestamp')
            split_idx = int(len(user_data) * train_ratio)
            
            train_data[user]['data'] = user_data.iloc[:split_idx]
            test_data[user]['data'] = user_data.iloc[split_idx:]
            
            train_data[user]['trajectories'] = self.create_trajectories(train_data[user]['data'])
            test_data[user]['trajectories'] = self.create_trajectories(test_data[user]['data'])
        
        return train_data, test_data
    
    def create_trajectories(self, user_data, sequence_length=5):
        trajectories = []
        data = user_data.values.tolist()
        
        for i in range(len(data) - sequence_length):
            trajectory = data[i:i+sequence_length]
            next_poi = data[i+sequence_length][1]
            trajectories.append((trajectory, next_poi))
            
        return trajectories