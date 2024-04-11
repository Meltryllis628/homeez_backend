import json
import csv
import os
from pathlib import Path
import ast
import shutil

import numpy as np
import math
import re
import random
import os

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.dataloader import default_collate
from torch.distributions.categorical import Categorical
from torchvision import transforms

import logging

from .model import *
from .dataset import *

furn_csv = os.path.join(os.getcwd(), 'furniture.csv')
furn_list = ['Basin',
             'Bed',
             'Bed Head', 
             'Bed Platform',
             'Bed Storage',
             'Bedside Table',
             'Coffee Table',
             'Dining Table',
             'Dressing Table',
             'Non Paid Study Table',
             'Dining Chair',
             'Sofa',
             'Feature Wall',
             'Fridge',
             'Settee',
             'Shower Screens',
             'Standing Lamp',
             'Study Table',
             'TV Console',
             'Table Lamp',
             'Toilet',
             'Toilet Bowl',
             'Toilet Sink',
             'Shoe Cabinet',
             'Storage Cabinets',
             'Tall Cabinet',
             'Bottom Island Kitchen Cabinet',
             'Bottom Kitchen Cabinet',
             'Bottom Vanity Cabinet',
             'Top Kitchen Cabinet',
             'Top Vanity Cabinet Mirror',
             'Top Vanity Cabinet Storage',
             'Wardrobes']
room_list = ['bath', 'bedroom', 'kitchen', 'living']
furn_cat = []
save_path = 'cvae_model2_2.pth'
with open(furn_csv, newline='') as f:
    csv_file = csv.DictReader(f)
    for furn in csv_file:
        furn_cat.append(furn['model_name'])
transform = transforms.Compose([
    transforms.ToTensor(),
])
device = torch.device("cpu")
model = FurniturePlacementModel(num_room_types=len(room_list), num_furniture_names=len(furn_list), num_furniture_types=len(furn_cat)).to(device)
model.load_state_dict(torch.load(save_path))
model.eval()  # Set the model to evaluation mode

def get_furn_colors(furn_list):
    # Categorize furniture by function
    categories = {
        'Table': ['Bedside Table', 'Coffee Table', 'Dining Table', 'Dressing Table', 'Non Paid Study Table', 'Study Table'],
        'Bed': ['Bed', 'Bed Head', 'Bed Platform', 'Bed Storage'],
        'Storage': ['Shoe Cabinet', 'Storage Cabinets', 'Tall Cabinet', 'Wardrobes',
                    'Bottom Island Kitchen Cabinet', 'Bottom Kitchen Cabinet', 'Bottom Vanity Cabinet'],
        'Storage2': ['Top Kitchen Cabinet', 'Top Vanity Cabinet Storage'],
        'Seating': ['Dining Chair'],
        'Seating2': ['Sofa'],
        'Toilet': ['Toilet'],
        'Toilet2': ['Toilet Bowl'],
        'Sink': ['Basin', 'Toilet Sink'],
        'Lighting': ['Standing Lamp', 'Table Lamp'],
        'Other': ['Feature Wall', 'Fridge', 'Shower Screens', 'TV Console', 'Top Vanity Cabinet Mirror', 'Settee']
    }

#     # Generate a colormap with distinct colors for each category
#     cmap = plt.cm.get_cmap('hsv', len(categories))

#     # Map each category to a color
#     category_colors = {category: cmap(i)[:3] for i, category in enumerate(categories)}

    # rgb values from https://www.rapidtables.com/web/color/RGB_Color.html
    category_colors = {
        'Table': (165, 42, 42), # brown
        'Bed': (240, 230, 140),  # khaki
        'Storage': (210, 105, 30), # chocolate
        'Storage2': (0, 139, 139), # dark cyan
        'Seating': (64, 224, 208), # turquoise
        'Seating2': (147, 112, 219), # medium purple
        'Toilet': (46, 139, 87), # sea green
        'Toilet2': (240, 128, 128), # light coral
        'Sink': (238, 130, 238), # violet
        'Lighting': (255, 215, 0), # gold
        'Other': (176,196,222) # light steel blue
    }

    # Map each furniture to its category color
    furniture_colors = {}
    for category, items in categories.items():
        for item in items:
            # Convert color to RGB values in the range [0, 255]
            furniture_colors[item] = category_colors[category]

    # Handle any furniture items not explicitly listed in the categories
    uncategorized_color = (128, 128, 128)  # Gray color for uncategorized items
    for furn in furn_list:
        if furn not in furniture_colors:
            furniture_colors[furn] = uncategorized_color

    return furniture_colors

def draw_room(state, output_path, furn_color, draw_furniture=False, save=False, furn_key='furniture'):
    # Define image size (assume square image)
    img_size = (200, 200)

    # Create a blank image
    img = Image.new('RGB', img_size, 'white')
    draw = ImageDraw.Draw(img)

    scale_factor_x = state['img_scale_factor_x']
    scale_factor_y = state['img_scale_factor_y']
    offset_x = state['img_offset_x']
    offset_y = state['img_offset_y']

    # Draw lines between room corners
    corners = [(round((corner[0] - offset_x) / scale_factor_x) , round((corner[1] - offset_y) / scale_factor_y)) for corner in state['room_corners']]
    corners.append(corners[0])
    draw.line(corners, fill='black', width=2)

    # Draw doors and windows
    for door in state['doors']:
        door_x = round((door['x'] - offset_x) / scale_factor_x)
        door_y = round((door['y'] - offset_y) / scale_factor_y)
        door_x2 = round(door['width'] / scale_factor_x)
        door_y2 = round(door['height'] / scale_factor_y)
        draw.rectangle([(door_x, door_y), (door_x + door_x2, door_y + door_y2)], fill='brown')

    for window in state['windows']:
        window_x = round((window['x'] - offset_x) / scale_factor_x)
        window_y = round((window['y'] - offset_y) / scale_factor_y)
        window_x2 = round(window['width'] / scale_factor_x)
        window_y2 = round(window['height'] / scale_factor_y)
        draw.rectangle([(window_x, window_y), (window_x + window_x2, window_y + window_y2)], fill='blue')

    # Draw furniture (for visualization only)
    if draw_furniture:
        for furniture in state[furn_key]:
            # Get the color assigned to this type of furniture
            color = furn_color[furniture['name']]

            # Draw the furniture as a rectangle (adjust as needed)
            x = round((furniture['scale_x'] - offset_x) / scale_factor_x)
            y = round((furniture['scale_y'] - offset_y) / scale_factor_y)
            length = round(furniture['scale_length'] / scale_factor_x) 
            width = round(furniture['scale_width'] / scale_factor_y)

            # rotation
            if furniture['rotation'] == 90 or furniture['rotation'] == 270:
                length = round(furniture['scale_width'] / scale_factor_x)
                width = round(furniture['scale_length'] / scale_factor_y)

            draw.rectangle([(x - length/2, y - width/2), (x + length/2, y + width/2)], fill=color)

            # Draw rotation indicator (line)
            end_x = x
            end_y = y
            if furniture['rotation'] == 0:
                end_x = x + length/2
            if furniture['rotation'] == 90:
                end_y = y + width/2
            if furniture['rotation'] == 180:
                end_x = x - length/2
            if furniture['rotation'] == 270:
                end_y = y - width/2
            draw.line([(x, y), (end_x, end_y)], fill='black', width=1)

    if save:
        img.save(output_path)
    
    return img

def visualize_room(filtered_name_classes, filtered_type_classes, filtered_coords, filtered_rot_classes, furn_list, furn_cat, state, furn_color):
    new_state = state.copy()
    new_state['furniture'] = []
    
    for i in range(len(filtered_name_classes)):
        furniture_name_index = filtered_name_classes[i]
        furniture_type_index = filtered_type_classes[i]

        # Ignore if furniture padding
        if furniture_name_index == len(furn_list) or furniture_type_index == len(furn_cat):
            continue

        x, y = filtered_coords[i]
        rotation = filtered_rot_classes[i]

        # Get furniture name and type from indices
        furniture_name = furn_list[furniture_name_index]
        furniture_type = furn_cat[furniture_type_index]

        # Placeholder for furniture dimensions - you may want to include actual dimensions in your model or have a lookup table
        length, width = 15, 10  # Placeholder dimensions

        new_furniture = {
            'name': furniture_name,
            'type': furniture_type,
            'scale_x': (x * state['scale_factor_x']) + state['offset_x'],
            'scale_y': (y * state['scale_factor_y']) + state['offset_y'],
            'rotation': rotation * 90,
            'scale_length': length,
            'scale_width': width,
        }
        
        #print(new_furniture)

        new_state['furniture'].append(new_furniture)
    
    return draw_room(new_state, "", furn_color, draw_furniture=True, save=False)

def is_outside(test_x, test_y, xs, ys):
    xs = xs.copy()
    ys = ys.copy()
    xs.append(xs[0])
    ys.append(ys[0])
    
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    
    in_room_boundary = min_x <= test_x and test_x <= max_x and min_y <= test_y and test_y <= max_y

    # for all walls directly perpendicular to test_x (between x_min and x_max), sort y values from smallest to largest
    # create tuplets (y1, y2), (y3, y4), ...
    # if test_y not between any y values in tuplet, is outside
    # same for test_y
    
    wall_x = []
    wall_y = []
    for i in range(len(xs) - 1):
        cmin_x = min(xs[i], xs[i + 1])
        cmax_x = max(xs[i], xs[i + 1])
        cmin_y = min(ys[i], ys[i + 1])
        cmax_y = max(ys[i], ys[i + 1])

        if abs(cmax_y - cmin_y) > abs(cmax_x - cmin_x) and cmin_y <= test_y and test_y <= cmax_y:
            wall_x.append((cmin_x + cmax_x) / 2)
        if abs(cmax_x - cmin_x) > abs(cmax_y - cmin_y) and cmin_x <= test_x and test_x <= cmax_x:
            wall_y.append((cmin_y + cmax_y) / 2)

    within_x = False
    within_y = False
    
    for i in range(len(wall_y) // 2):
        min_wall_y = min(wall_y[2 * i], wall_y[2 * i + 1])
        max_wall_y = max(wall_y[2 * i], wall_y[2 * i + 1])
        if min_wall_y <= test_y and test_y <= max_wall_y:
            within_y = True
            break
    for i in range(len(wall_x) // 2):
        min_wall_x = min(wall_x[2 * i], wall_x[2 * i + 1])
        max_wall_x = max(wall_x[2 * i], wall_x[2 * i + 1])
        if min_wall_x <= test_x and test_x <= max_wall_x:
            within_x = True
            break

    return not in_room_boundary or not (within_x and within_y)
def post_process(filtered_name_classes, filtered_type_classes, filtered_coords, filtered_rot_classes, json_file):
    with open(json_file, 'r') as f:
        state = json.load(f)
    corners = [corner for corner in state['norm_room_corners']]
    xs = [s[0] for s in state['norm_room_corners']]
    ys = [s[1] for s in state['norm_room_corners']]

    new_filtered_name_classes = []
    new_filtered_type_classes = []
    new_filtered_coords = []
    new_filtered_rot_classes = []
    
    for i, c in enumerate(filtered_coords):
        x, y = c
        if not is_outside(x, y, xs, ys) and x > 0.05 and y > 0.05 and c not in filtered_coords[:i]:
            new_filtered_name_classes.append(filtered_name_classes[i])
            new_filtered_type_classes.append(filtered_type_classes[i])
            new_filtered_coords.append(filtered_coords[i])
            new_filtered_rot_classes.append(filtered_rot_classes[i])

    return new_filtered_name_classes, new_filtered_type_classes, new_filtered_coords, new_filtered_rot_classes

def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def check_bad_layout(filtered_name_classes, filtered_coords, json_file, furn_list, room_type):
    with open(json_file, 'r') as f:
        state = json.load(f)

    # **GENERAL CHECK**
    # check if too little furniture is present
    if len(filtered_name_classes) < 2:
        return True
    
    # check if furniture is too close to a door
    for i in range(len(filtered_name_classes)):
        furniture_name = furn_list[filtered_name_classes[i]]
        x, y  = filtered_coords[i]
        if furniture_name != 'Standing Lamp':
            for door in state['doors']:
                if dist(door['norm_x'], door['norm_y'], x, y) < 0.2:
                    return True
                    
    # **ROOM SPECIFIC CHECK**
    # check for unusual number of furniture
    num_tvs = 0
    num_fridge = 0
    
    for i in range(len(filtered_name_classes)):
        furniture_name = furn_list[filtered_name_classes[i]]
        if furniture_name == 'TV Console':
            num_tvs += 1
        if furniture_name == 'Fridge':
            num_fridge += 1
    
    if room_type == 'living' and num_tvs > 1:
        return True
    if room_type == 'kitchen' and num_fridge == 0:
        return True
    
    # check if toilet and basin is too close
    if room_type == 'bath':
        a = []
        b = []
        
        for i in range(len(filtered_name_classes)):
            furniture_name = furn_list[filtered_name_classes[i]]
            x, y  = filtered_coords[i]
    
            if 'Toilet' in furniture_name:
                a.append((x, y))
            if 'Basin' in furniture_name:
                b.append((x, y))

        for a_n in a:
            for b_n in b:
                if dist(a_n[0], a_n[1], b_n[0], b_n[1]) < 0.3:
                    return True
    
    # check if table and chair is too far
    if room_type == 'living':
        a = []
        b = []
        
        for i in range(len(filtered_name_classes)):
            furniture_name = furn_list[filtered_name_classes[i]]
            x, y  = filtered_coords[i]

            if 'Chair' in furniture_name or 'Sofa' in furniture_name:
                a.append((x, y))

            if 'Table' in furniture_name:
                b.append((x, y))

        for a_n in a:
            is_far = True
            for b_n in b:
                if dist(a_n[0], a_n[1], b_n[0], b_n[1]) < 0.2:
                    is_far = False
                    break
            if is_far:
                return True
        
    return False

def get_output(json_file, original_state, type_to_url, names, types, coords, rotation, output_dir, file_name):
    with open(json_file, 'r') as f:
        state = json.load(f)

    new_state = state.copy()
    new_state['furniture'] = []

    with open(type_to_url, 'r') as f:
        mapping = json.load(f)

    scale_x = original_state['scale_factor_x']
    scale_y = original_state['scale_factor_y']
    offset_x = original_state['offset_x']
    offset_y = original_state['offset_y']

    for i in range(len(names)):
        norm_x = coords[i][0]
        norm_y = coords[i][1]

        scaled_x = norm_x * scale_x + offset_x
        scaled_y = norm_y * scale_y + offset_y

        actual_x = scaled_x * 25 / 1000
        actual_y = scaled_y * 25 / 1000
        
        new_furn = {
            'name': names[i],
            'type': types[i],
            'norm_x': coords[i][0],
            'norm_y': coords[i][1],
            'x': round(scaled_x),
            'y': round(scaled_y),
            'coordinates':{
                'x': actual_x,
                'y': 0,
                'z': actual_y
            },
            'rotY': rotation[i] * 90 * math.pi / 180,
            'url': mapping[types[i]][:-4],  # remove .glb
            'price': 'Non Paid Item',
        }

        new_state['furniture'].append(new_furn)

    out_path = f'{output_dir}/{os.path.basename(file_name)}'
    
    with open(out_path, 'w') as output:
        json.dump(new_state, output)

def generate(test_room, image_directory, image_dest_directory, type_to_url, output_directory, save_path):
    dataset = FurnitureDataset(json_files=[test_room], img_dir=image_directory, room_list=room_list, furn_list=furn_list, furn_cat=furn_cat, is_gen=True, transform=transform)

    test_dataloader2 = DataLoader(dataset, batch_size=1, shuffle=False)

    with open(test_room, 'r') as f:
        state = json.load(f)

    # for furn in state['furniture']:
        #print(furn, '\n')
        
    furn_color = get_furn_colors(furn_list)
    draw_room(state, "", furn_color, draw_furniture=False, save=False)
    model = FurniturePlacementModel(num_room_types=len(room_list), num_furniture_names=len(furn_list), num_furniture_types=len(furn_cat)).to(device)
    model.load_state_dict(torch.load(save_path))
    model.eval()  # Set the model to evaluation mode
    # the lower this value, the more furniture will be generated (with a lower confidence score)
    conf_threshold = 0.9

    with torch.no_grad():
        for i, data in enumerate(test_dataloader2):
            image, room_features_tensor, room_type_tensor, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor, _, json_file = data

            # Input features
            image = image.to(device)
            room_features_tensor = room_features_tensor.to(device)
            room_type_tensor = room_type_tensor.to(device)
            furniture_features_tensor = furniture_features_tensor.to(device)
            furniture_names_tensor = furniture_names_tensor.to(device)
            furniture_types_tensor = furniture_types_tensor.to(device)
            placed_furniture_tensor = placed_furniture_tensor.to(device)

            # Labels
            filtered_name_classes, filtered_type_classes, filtered_coords, filtered_rot_classes = model(image, room_features_tensor, room_type_tensor, furniture_features_tensor, furniture_names_tensor, furniture_types_tensor, placed_furniture_tensor, conf_threshold=conf_threshold, deterministic=True)

            if filtered_name_classes.shape[0] > 1:
                filtered_name_classes = filtered_name_classes.squeeze(0)
                filtered_type_classes = filtered_type_classes.squeeze(0)
                filtered_coords = filtered_coords.squeeze(0)
                filtered_rot_classes = filtered_rot_classes.squeeze(0)

            filtered_name_classes = filtered_name_classes.tolist()
            filtered_type_classes = filtered_type_classes.tolist()
            filtered_coords = filtered_coords.tolist()
            filtered_rot_classes = filtered_rot_classes.tolist()

            # Filter out abnormal furniture placements
            filtered_name_classes, filtered_type_classes, filtered_coords, filtered_rot_classes = post_process(filtered_name_classes, filtered_type_classes, filtered_coords, filtered_rot_classes, json_file[i])
            room_type = room_list[room_type_tensor.item()]

            # Remove room if layout is 'bad'
            if check_bad_layout(filtered_name_classes, filtered_coords, json_file[i], furn_list, room_type):
                #print('Skipped layout')
                continue
            
            #print(f"Room Type: {room_type}")
            furniture_names = [furn_list[furniture_name] for furniture_name in filtered_name_classes]
            #print(furniture_names)
            
            img = visualize_room(filtered_name_classes, filtered_type_classes, filtered_coords, filtered_rot_classes, furn_list, furn_cat, state, furn_color)
            furniture_types = [furn_cat[furniture_type] for furniture_type in filtered_type_classes]
            
            #plt.title(f'{room_type}: {json_file[i]}')
            #plt.imshow(img)
            #plt.savefig(f'{image_dest_directory}/{Path(json_file[i]).stem}.png')

            get_output(json_file[i], state, type_to_url, furniture_names, furniture_types, filtered_coords, filtered_rot_classes, output_directory, test_room)

def generate_outputs(input_jsons, image_directory, image_dest_directory,  output_directory):
    if not os.path.exists(image_dest_directory):
        os.makedirs(image_dest_directory)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    save_path = 'cvae_model2_2.pth'
    type_to_url = 'type_to_url.json'
    for json_file in input_jsons:
        generate(json_file, image_directory, image_dest_directory, type_to_url, output_directory, save_path)