import json
import os
from pathlib import Path

from PIL import Image

import torch

from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate

class FurnitureDataset(Dataset):
    def __init__(self, json_files, img_dir, room_list, furn_list, furn_cat, is_gen=False, transform=None):
        self.json_files = json_files
        self.img_dir = img_dir
        self.room_list = room_list
        self.furn_list = furn_list
        self.furn_cat = furn_cat
        self.is_gen = is_gen
        self.transform = transform
        self.max_corners = 25
        self.max_windows = 10
        self.max_doors = 10
        self.max_furniture = 15
        
    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_file = self.json_files[idx]
        with open(json_file, 'r') as f:
            state = json.load(f)

        # Image tensor
        img_base_name = Path(json_file).stem
        if self.is_gen:
            img_base_name = img_base_name.split('_')[0]
        img_name = os.path.join(self.img_dir, f"{img_base_name}.png")
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # Room type tensor
        room_type = self.room_list.index(state['room_type'])
        room_type_tensor = torch.tensor([room_type], dtype=torch.long)

        # Room features tensor
        corners = [corner for corner in state['norm_room_corners']] + [[0, 0]] * (self.max_corners - len(state['norm_room_corners']))
        doors = [[door['norm_x'], door['norm_y'], door['norm_width'], door['norm_height']] for door in state['doors']] + [[0, 0, 0, 0]] * (self.max_doors - len(state['doors']))
        windows = [[window['norm_x'], window['norm_y'], window['norm_width'], window['norm_height']] for window in state['windows']] + [[0, 0, 0, 0]] * (self.max_windows - len(state['windows']))
        
        corners_tensor = torch.tensor(corners, dtype=torch.float32)
        corners_tensor = torch.flatten(corners_tensor)
        
        doors_tensor = torch.tensor(doors, dtype=torch.float32)
        doors_tensor = torch.flatten(doors_tensor)
        
        windows_tensor = torch.tensor(windows, dtype=torch.float32)
        windows_tensor = torch.flatten(windows_tensor)
            
        room_features_tensor = torch.cat([corners_tensor, doors_tensor, windows_tensor])

        # Furniture names tensor (generic furniture label, e.g. table, chair)
        furniture_names = [self.furn_list.index(furn['name']) for furn in state['furniture']] + [len(self.furn_list)] * (self.max_furniture - len(state['furniture']))
        furniture_names_tensor = torch.tensor(furniture_names, dtype=torch.long)
        
        # Furniture types tensor (actual furniture model)
        furniture_types = [self.furn_cat.index(furn['type']) for furn in state['furniture']] + [len(self.furn_cat)] * (self.max_furniture - len(state['furniture']))
        furniture_types_tensor = torch.tensor(furniture_types, dtype=torch.long)

        # Furniture features tensor
        furniture_features = [[furn['norm_length'],
                              furn['norm_width']
                              ] for furn in state['furniture']] + [[0, 0]] * (self.max_furniture - len(state['furniture']))
        furniture_features_tensor = torch.tensor(furniture_features, dtype=torch.float32)

        # Placed furniture tensor
        placed_furniture = [[
            self.furn_list.index(furn['name']),
            self.furn_cat.index(furn['type']),
            furn['norm_x'],
            furn['norm_y'],
            furn['rotation'] // 90
        ] for furn in state['placed']] + [[len(self.furn_list), len(self.furn_cat), 0, 0, 0]] * (self.max_furniture - len(state['placed']))
        placed_furniture_tensor = torch.tensor(placed_furniture, dtype=torch.float32)
        
        # Furniture details tensor for evaluation
        if self.is_gen:
            furniture_details = [[
                self.furn_list.index(furn['name']),
                self.furn_cat.index(furn['type']),
                0,
                0,
                0
            ] for furn in state['furniture']] + [[len(self.furn_list), len(self.furn_cat), 0, 0, 0]] * (self.max_furniture - len(state['furniture']))
        else:
            furniture_details = [[
                self.furn_list.index(furn['name']),
                self.furn_cat.index(furn['type']),
                furn['norm_x'],
                furn['norm_y'],
                furn['rotation'] // 90
            ] for furn in state['furniture']] + [[len(self.furn_list), len(self.furn_cat), 0, 0, 0]] * (self.max_furniture - len(state['furniture']))

        furniture_details_tensor = torch.tensor(furniture_details, dtype=torch.float32)

        return image, room_features_tensor, room_type_tensor, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor, furniture_details_tensor, json_file
    
def collate_fn(batch):
    # Initialize empty lists for each type of data
    images, room_features_tensors, room_type_tensors, furniture_features_tensors, furniture_names_tensors, furniture_types_tensors, placed_furniture_tensors, furniture_details_tensors = [], [], [], [], [], [], [], []
    json_files = []

    for items in batch:
        # Unpack all tensors and the state dictionary for each item
        image, room_features_tensor, room_type_tensor, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor, furniture_details_tensor, json_file = items
        
        # Append tensors to their respective lists
        images.append(image)
        room_features_tensors.append(room_features_tensor)
        room_type_tensors.append(room_type_tensor)
        furniture_features_tensors.append(furniture_features_tensor)
        furniture_names_tensors.append(furniture_names_tensor)
        furniture_types_tensors.append(furniture_types_tensor)
        placed_furniture_tensors.append(placed_furniture_tensor)
        furniture_details_tensors.append(furniture_details_tensor)

        # Append the json_file to the list
        json_files.append(json_file)

    # Collate each list of tensors using default_collate
    images = default_collate(images)
    room_features_tensors = default_collate(room_features_tensors)
    room_type_tensors = default_collate(room_type_tensors)
    furniture_features_tensors = default_collate(furniture_features_tensors)
    furniture_names_tensors = default_collate(furniture_names_tensors)
    furniture_types_tensors = default_collate(furniture_types_tensors)
    placed_furniture_tensors = default_collate(placed_furniture_tensors)
    furniture_details_tensors = default_collate(furniture_details_tensors)

    # Combine all collated tensors and the list of state dictionaries into the final batch
    final_batch = (images, room_features_tensors, room_type_tensors, furniture_features_tensors, furniture_names_tensors, furniture_types_tensors, placed_furniture_tensors, furniture_details_tensors, json_files)

    return final_batch 