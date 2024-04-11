import json
import os
from pathlib import Path
import numpy as np
import cv2
import re
import ast
import csv
import math
from PIL import Image, ImageDraw
from itertools import chain, combinations
import random
BEDROOM_STARTER_PACK = [
    {
        "name": "Wardrobes",
        "type": "COH_BWC_MV1",
    },
    {
        "name": "Bed",
        "type": "BK_Bed_1",
    },
]
KITCHEN_STARTER_PACK = [
    {
        "name": "Fridge",
        "type": "BK_Fridge_5",
    },
    {
        "name": "Top Kitchen Cabinet",
        "type": "HOM_THCabinet_MV1_1",
    },
    {
        "name": "Bottom Kitchen Cabinet",
        "type": "HOM_BHCabinet_MV3_5",
    }
]
BATH_STARTER_PACK = [
    {
        "name": "Basin",
        "type": "BK_TSink_1",
    },
    {
        "name": "Toilet",
        "type": "BK_Toilet_1",
    },
]
LIVING_ROOM_STARTER_PACK = [
    {
        "name": "TV Console",
        "type": "COH_TVCS_B_MMV1",
    },
    {
        "name": "Sofa",
        "type": "COH_S_2S_1",
    },
    {
        "name": "Dining Chair",
        "type": "COH_DC_MV1",
    },
    {
        "name": "Dining Chair",
        "type": "COH_DC_MV1",
    },
    {
        "name": "Dining Table",
        "type": "COH_DT_MV2_4P",
    },
    {
        "name": "Coffee Table",
        "type": "COH_CT_MV2",
    }
]
def process_url(url):
    regex = r'([^/]+)\.glb$'
    match = re.search(regex, url)
    if match:
        furn_name = match.group(1)
        return furn_name
    return None

def load_furn_catalog(furn_cat_file): 
    furns = {}
    with open(furn_cat_file, newline='') as csv_file:
        furn_cat = csv.DictReader(csv_file)

        for furn in furn_cat:
            furns[furn['model_name']] = ast.literal_eval(furn['size'])

    return furns
def rect_in_polygon(rect, polygon):
    x, y, w, h = rect
    for i in range(x, x+w):
        for j in range(y, y+h):
            if cv2.pointPolygonTest(polygon, (i, j), False) < 0:
                return False
    return True

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    l = list(chain.from_iterable(combinations(s, r) for r in range(len(s)+1)))
    l.pop()
    
    return l

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

    # Draw furniture
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
                length = round(furniture['scale_width'] / scale_factor_y)
                width = round(furniture['scale_length'] / scale_factor_x)

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
furniture_list = ['Basin', 'Toilet', 'Sofa', 'Fridge', 'TV Console', 'Wardrobes', 'Bottom Kitchen Cabinet', 'Dining Table', 'Dining Chair', 'Coffee Table', 'Bed Storage', 'Bed Platform', 'Storage Cabinets', 'Non Paid Study Table', 'Study Table', 'Top Kitchen Cabinet', 'Standing Lamp', 'Bottom Island Kitchen Cabinet', 'Bed', 'Shoe Cabinet', 'Top Vanity Cabinet Mirror', 'Dressing Table', 'Tall Cabinet', 'Feature Wall', 'Bottom Vanity Cabinet', 'Toilet Sink', 'Toilet Bowl', 'Top Vanity Cabinet Storage', 'Bedside Table', 'Table Lamp', 'Shower Screens', 'Bed Head', 'Settee']
furniture_size_csv = 'furniture_size.csv'

def parse_input_json(input_file_path, furniture_list = furniture_list, furniture_size_csv = furniture_size_csv):
    input_file_name = os.path.basename(input_file_path).split('.')[0]
    raw_json_object = json.load(open(input_file_path))
    rooms = raw_json_object['rooms']
    num_rooms = {}
    output_dir = Path(input_file_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    json_output_dir = os.path.join(output_dir, 'json_raw')
    img_output_dir = os.path.join(output_dir, 'img')
    if not os.path.exists(json_output_dir):
        os.makedirs(json_output_dir)
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)
    for room in rooms:
        if room["category"] not in ["living", "kitchen", "bath", "bedroom"]:
            continue
        json_dict = {'room_corners':[], 'doors':[], 'windows':[], 'furniture':[]}
        json_dict['room_type'] = room['category']
        category = room['category']
        if category not in num_rooms:
            num_rooms[category] = 0
        num_rooms[category] += 1
        xs = []
        ys = []
        for x, y in zip(room['xs'], room['ys']):
            x = int(x)
            y = int(y)
            json_dict['room_corners'].append([x, y])
            xs.append(x)
            ys.append(y)
        corners = np.array(list(zip(xs, ys))) # 原始输入，不能动的
        img = np.zeros((1000, 1000, 3), np.uint8)
        cv2.fillPoly(img, [corners], (255, 255, 255))
        # 膨胀
        kernel = np.ones((11,11), np.uint8)
        img = cv2.dilate(img, kernel, iterations=1)
        # 二值化
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, 0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0]

        for door in raw_json_object['doors']:
            rect = (door["x"], door["y"], door["width"], door["height"])
            if rect_in_polygon(rect, contours):
                json_dict['doors'].append(door)

        for window in raw_json_object['windows']:
            rect = (window["x"], window["y"], window["width"], window["height"])
            if rect_in_polygon(rect, contours):
                json_dict['windows'].append(window)

        # Furniture Starter Pack
        # if category == 'living':
        #     json_dict['furniture'] = LIVING_ROOM_STARTER_PACK
        # elif category == 'kitchen':
        #     json_dict['furniture'] = KITCHEN_STARTER_PACK
        # elif category == 'bath':
        #     json_dict['furniture'] = BATH_STARTER_PACK
        # elif category == 'bedroom':
        #     json_dict['furniture'] = BEDROOM_STARTER_PACK

        max_x, min_x, max_y, min_y = max(xs), min(xs), max(ys), min(ys)
        room_length = max_x - min_x
        room_width = max_y - min_y

        json_dict['scale_factor_x'] = room_length
        json_dict['scale_factor_y'] = room_width
        json_dict['offset_x'] = min_x
        json_dict['offset_y'] = min_y

        img_size = (200,200)

        json_dict['img_scale_factor_x'] = room_length / img_size[0] / 0.9   # margin of 0.1
        json_dict['img_scale_factor_y'] = room_width / img_size[1] / 0.9
        json_dict['img_offset_x'] = min_x - room_length * 0.05   # center the offsets
        json_dict['img_offset_y'] = min_y - room_width * 0.05

        json_dict['norm_room_corners'] = []
        for i in range(len(xs)):
            offset_coords = ((xs[i] - json_dict['offset_x']) / json_dict['scale_factor_x'],
                             (ys[i] - json_dict['offset_y']) / json_dict['scale_factor_y'])
            json_dict['norm_room_corners'].append(offset_coords)

        for door in json_dict['doors']:
            door['norm_x'] = (door['x'] - json_dict['offset_x']) / json_dict['scale_factor_x']
            door['norm_y'] = (door['y'] - json_dict['offset_y']) / json_dict['scale_factor_y']
            door['norm_width'] = door['width'] / json_dict['scale_factor_x']
            door['norm_height'] = door['height'] / json_dict['scale_factor_y']
        for window in json_dict['windows']:
            window['norm_x'] = (window['x'] - json_dict['offset_x']) / json_dict['scale_factor_x']
            window['norm_y'] = (window['y'] - json_dict['offset_y']) / json_dict['scale_factor_y']
            window['norm_width'] = window['width'] / json_dict['scale_factor_x']
            window['norm_height'] = window['height'] / json_dict['scale_factor_y'] 

        furn_cat = load_furn_catalog(furniture_size_csv)

        furniture_for_new_room = []
        for furniture in json_dict['furniture']:
            empty_furniture = {}
            furniture_model_name = furniture["type"]
            empty_furniture['type'] = furniture_model_name
            empty_furniture['name'] = furniture["name"]
            length = furn_cat[furniture_model_name][0]
            width = furn_cat[furniture_model_name][2]

            # x and y are in meters
            # length and width are in mm
            scale_length = length / 25
            scale_width = width / 25
            rotation = 0

            x = np.random.uniform(min_x + scale_length/2, max_x - scale_length/2)
            y = np.random.uniform(min_y + scale_width/2, max_y - scale_width/2)
            
            norm_x = (x - json_dict['offset_x']) / json_dict['scale_factor_x']
            norm_y = (y - json_dict['offset_y']) / json_dict['scale_factor_y']
            norm_length = scale_length / json_dict['scale_factor_x']
            norm_width = scale_width / json_dict['scale_factor_y']
            
            
            furniture['length'] = length
            furniture['width'] = width
            
            furniture['scale_x'] = x
            furniture['scale_y'] = y
            furniture['scale_length'] = scale_length
            furniture['scale_width'] = scale_width
            furniture['rotation'] = rotation
            
            furniture['norm_x'] = norm_x
            furniture['norm_y'] = norm_y
            furniture['norm_length'] = norm_length
            empty_furniture['norm_length'] = norm_length
            furniture['norm_width'] = norm_width   
            empty_furniture['norm_width'] = norm_width
            furniture_for_new_room.append(empty_furniture)

        all_furn_pos = powerset(json_dict['furniture'])
        furniture_color_list = get_furn_colors(furniture_list)

        # For empty room
        new_state = json_dict.copy()
        new_state['placed'] = []

        room_name = category + str(num_rooms[category])
        
        json_output_path = os.path.join(json_output_dir, room_name + '.json')
        with open(json_output_path, 'w') as output_f:
            json.dump(new_state, output_f)

        img_output_path = os.path.join(img_output_dir, room_name + '.png')
        draw_room(new_state, img_output_path, furniture_color_list, draw_furniture=False, save=True)

        # For all possible subsets of furniture placements
        for i, pos in enumerate(all_furn_pos):
            new_state = json_dict.copy()
            new_state['placed'] = []
            
            for furn in pos:
                new_state['placed'].append(furn)
                json_output_path = os.path.join(json_output_dir, room_name + '_' + str(i) + '.json')
                img_output_path = os.path.join(img_output_dir, room_name + '_' + str(i) + '.png')
                draw_room(new_state, img_output_path, furniture_color_list, draw_furniture=True, save=True, furn_key='placed')
                # out_json_obj = new_state.copy()
                # new_furnitures = []
                # for furniture in out_json_obj['placed']:
                #     new_furn = {}
                #     new_furn['type'] = furniture['type']
                #     new_furn['name'] = furniture['name']
                #     new_furn['norm_length'] = furniture['norm_length']
                #     new_furn['norm_width'] = furniture['norm_width']
                #     new_furnitures.append(new_furn)
                # out_json_obj['placed'] = new_furnitures
                # out_json_obj['furiniture'] = new_furnitures
                with open(json_output_path, 'w') as output:
                    json.dump(new_state, output)

        # For room with complete furniture
        # img_output_path = img_output_dir + room_name + '.png'
        # draw_room(json_dict, img_output_path, furniture_color_list, draw_furniture=True, save=True)
        # out_json_obj = json_dict.copy()
        # new_furnitures = []
        # for furniture in out_json_obj['furniture']:
        #     new_furn = {}
        #     new_furn['type'] = furniture['type']
        #     new_furn['name'] = furniture['name']
        #     new_furn['norm_length'] = furniture['norm_length']
        #     new_furn['norm_width'] = furniture['norm_width']
        #     new_furnitures.append(new_furn)
        # out_json_obj['placed'] = new_furnitures
        # out_json_obj['furiniture'] = new_furnitures
        # json_dict['placed'] = json_dict['furniture']
        # json_output_path = json_output_dir + room_name + '.json'
        # with open(json_output_path, 'w') as output:
        #     json.dump(json_dict, output)
    return output_dir

# main function, takes one argument which is the input json file

def generate_furniture(json_file, output_dir, image_output_dir, num_samples=5):
    furn_sets_file = 'furn_sets.json'
    num_furns = {
    'bath': [2, 3],
    'bedroom': [2, 3, 4],
    'kitchen': [2, 3, 4],
    'living': [7, 8, 9, 10]
    }
    with open(furn_sets_file, 'r') as f:
        furn_sets = json.load(f)
    
    with open(json_file, 'r') as f:
        state = json.load(f)
    
    room_type = state['room_type']
    
    for num_furn in num_furns[room_type]:
        if str(num_furn) not in furn_sets[room_type]:
            continue
        sets = furn_sets[room_type][str(num_furn)]
    
        for i in range(num_samples):
            random_set = random.choice(sets)
            state_copy = state.copy()
            state_copy['furniture'] = random_set

    
            file_name = os.path.join(output_dir,f'{os.path.splitext(os.path.basename(json_file))[0]}_{num_furn}_{i}.json') 
            with open(file_name, 'w') as f:
                json.dump(state_copy, f)
            # furniture_color_list = get_furn_colors(furniture_list)
            # img_output_path = os.path.join(image_output_dir, f'{os.path.splitext(os.path.basename(json_file))[0]}_{num_furn}_{i}.png')
            # draw_room(state_copy, img_output_path, furniture_color_list, draw_furniture=True, save=True, furn_key='placed')