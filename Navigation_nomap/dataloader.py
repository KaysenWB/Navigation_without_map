
import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
from torchvision import transforms


# data format expected
"""
streetlearn_data/
├── shen_zhen/
│   ├── nodes/
│   │   ├── node_001/
│   │   │   ├── front.jpg    
│   │   │   ├── back.jpg     
│   │   │   ├── left.jpg     
│   │   │   └── right.jpg    
│   │   ├── node_002/
│   │   └── ...
│   ├── graph.json          # relationship of nodes
└── metadata.csv            # supply information, such as GPS
"""



class StreetLearnDataset(Dataset):
    def __init__(self, root_dir, city='shen_zhen', trans=None):
        self.root_dir = os.path.join(root_dir, city)
        self.transform = trans or self.default_transform()

        # 加载元数据
        self.metadata = pd.read_csv(os.path.join(root_dir, 'metadata.csv'))
        with open(os.path.join(self.root_dir, 'graph.json'), 'r') as f:
            self.graph = json.load(f)  # {node_id: [neighbor_ids]}

        self.node_ids = list(self.graph.keys())

    def default_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.node_ids)

    def __getitem__(self, idx):
        node_id = self.node_ids[idx]
        node_dir = os.path.join(self.root_dir, 'nodes', f'node_{node_id:03d}')

        views = {}
        for view_name in ['front', 'back', 'left', 'right']:
            img_path = os.path.join(node_dir, f'{view_name}.jpg')
            img = Image.open(img_path).convert('RGB')
            if self.transform:
                img = self.transform(img)
            views[view_name] = img

        gps = torch.tensor([
            self.metadata[self.metadata['node_id'] == node_id]['lat'].values[0],
            self.metadata[self.metadata['node_id'] == node_id]['lon'].values[0]
        ], dtype=torch.float)

        neighbors = self.graph[node_id]

        return {
            'views': views,
            'gps': gps,
            'node_id': node_id,
            'neighbors': neighbors
        }