import numpy as np
import matplotlib.pyplot as plt
import cv2
import json
import pandas as pd
import os

def draw_room(filepath, imgpath):
    sizes = pd.read_csv('furniture_size.csv')
    data = json.load(open(filepath))
    room_corners = np.array(data['room_corners'])
    x_min = room_corners[:, 0].min() - 10
    x_max = room_corners[:, 0].max() + 10
    y_min = room_corners[:, 1].min() - 10
    y_max = room_corners[:, 1].max() + 10

    img = np.zeros((y_max - y_min, x_max - x_min,3), dtype=np.uint8)
    img.fill(255)
    xs = []
    ys = []
    for corner in room_corners:
        corner[0] -= x_min
        xs.append(corner[0])
        corner[1] -= y_min
        ys.append(corner[1])
    # square lines

    corner_num = len(room_corners)
    for i in range(corner_num):
        cv2.fillPoly(img, [np.array(room_corners)], (0,0,0))
    #dilate
    kernel = np.ones((7,7),np.uint8)
    img0 = cv2.erode(img,kernel,iterations = 1)
    # take the difference
    img = 255 - ( img - img0)

    doors = data['doors']
    for door in doors:
        inside = ""
        x = door['x'] - x_min
        y = door['y'] - y_min
        w = door['width']
        h = door['height']
        if w<=h:
            cv2.rectangle(img, (x-8, y), (x+8, y+h), (255,255,255), -1)
        else:
            cv2.rectangle(img, (x, y-8), (x+w, y+8), (255,255,255), -1)
        if w < h:
            y = door['y'] - y_min + h//2
            l = door['x']-1 - x_min
            l_tmp = img[y, l, 0]
            count_l = 0
            r = door['x']+1 - x_min
            r_tmp = img[y, r, 0]
            count_r = 0
            while l > 0:
                l -= 1
                if img[y, l, 0] != l_tmp:
                    count_l += 1
                l_tmp = img[y, l, 0]
            while r < img.shape[1]-1:
                r += 1
                if img[y, r, 0] != r_tmp:
                    count_r += 1
                r_tmp = img[y, r, 0]
            if count_l > count_r:
                inside = "left"
            else:
                inside = "right"
        else:
            x = door['x'] - x_min + w//2
            u = door['y']-1 - y_min
            u_tmp = img[u, x, 0]
            count_u = 0
            d = door['y']+1 - y_min
            d_tmp = img[d, x, 0]
            count_d = 0
            while u > 0:
                u -= 1
                if img[u, x, 0] != u_tmp:
                    count_u += 1
                u_tmp = img[u, x, 0]
            while d < img.shape[0]-1:
                d += 1
                if img[d, x, 0] != d_tmp:
                    count_d += 1
                d_tmp = img[d, x, 0]
            if count_u > count_d:
                inside = "up"
            else:
                inside = "down"
        if inside == 'left':
            center_y = door['y'] - y_min
            center_x = door['x'] - x_min + w//2
            size = h
            cv2.line(img, (center_x, center_y), (center_x-size, center_y), (0,0,0), 1)
            # draw arc to indicate the door
            cv2.ellipse(img, (center_x, center_y), (size, size), 0, 90, 180, (0,0,0), 1)
        if inside == 'right':
            center_y = door['y'] - y_min
            center_x = door['x'] - x_min + w//2
            size = h
            cv2.line(img, (center_x, center_y), (center_x+size, center_y), (0,0,0), 1)
            # draw arc to indicate the door
            cv2.ellipse(img, (center_x, center_y), (size, size), 0, 0, 90, (0,0,0), 1)
        if inside == 'up':
            center_y = door['y'] - y_min + h//2
            center_x = door['x'] - x_min
            size = w
            cv2.line(img, (center_x, center_y), (center_x, center_y-size), (0,0,0), 1)
            # draw arc to indicate the door
            cv2.ellipse(img, (center_x, center_y), (size, size), 0, 270, 360, (0,0,0), 1)
        if inside == 'down':
            center_y = door['y'] - y_min + h//2
            center_x = door['x'] - x_min
            size = w
            cv2.line(img, (center_x, center_y), (center_x, center_y+size), (0,0,0), 1)
            # draw arc to indicate the door
            cv2.ellipse(img, (center_x, center_y), (size, size), 0, 0, 90, (0,0,0), 1)

    # HOM_PBedS_SV1_1,"(1003,1993)"

        
    windows = data['windows']
    for window in windows:
        x = window['x'] - x_min
        y = window['y'] - y_min
        w = window['width']
        h = window['height']
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,255,255), -1)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)

    furnitures = data['furniture']
    furnitures_new = []
    for furniture in furnitures:
        furniture_new = furniture.copy()
        # 'scale_x': (x * state['scale_factor_x']) + state['offset_x'],
        # 'scale_y': (y * state['scale_factor_y']) + state['offset_y'],
        x = furniture['x'] - x_min
        y = furniture['y'] - y_min
        size = sizes[sizes['model_name'] == furniture['type']]['size'].values[0]
        size = size.replace('(', '').replace(')', '').split(',')
        w = int(float(size[0])/25)
        h = int(float(size[2])/25)
        rotation = int(furniture["rotY"]/3.1415926*2)
        if rotation == 1 or rotation == 3:
            w, h = h, w
        x = x - w//2
        y = y - h//2
        furniture_new['x'] = x
        furniture_new['y'] = y
        furniture_new['w'] = w
        furniture_new['h'] = h
        furnitures_new.append(furniture_new)
    data['furniture'] = furnitures_new
    furnitures = data['furniture']
    furnitures_new = []

    for furniture in furnitures:
        furniture_new = furniture.copy()
        x = furniture['x']
        y = furniture['y']
        w = furniture['w']
        h = furniture['h']
        # print(furniture['name'], x, y, w, h)
        for door in doors:
            x2 = door['x'] - x_min
            y2 = door['y'] - y_min
            w2 = door['width'] if door['width'] > door['height'] else door['height']+2
            h2 = w2
            if x < x2+w2 and x+w > x2 and y < y2+h2 and y+h > y2:
                dx = 0
                dy = 0
                if x+w/2 < x2+w2/2:
                    dx = x2 - w - x
                else:
                    dx = x2 + w2 - x
                if y+h/2 < y2+h2/2:
                    dy = y2 - h - y
                else:
                    dy = y2 + h2 - y
                if abs(dx) < abs(dy):
                    x += dx
                else:
                    y += dy
        # print(furniture['name'], x, y, w, h)
        # STEP 1: If the furniture is in the area of another furniture or door openning space, move it to the nearest place
        for furniture2 in furnitures_new:
            if furniture2 == furniture:
                continue
            x2 = furniture2['x']
            y2 = furniture2['y']
            w2 = furniture2['w']
            h2 = furniture2['h']
            if x < x2+w2 and x+w > x2 and y < y2+h2 and y+h > y2:
                dx = 0
                dy = 0
                if x+w/2 < x2+w2/2:
                    dx = x2 - w - x
                else:
                    dx = x2 + w2 - x
                if y+h/2 < y2+h2/2:
                    dy = y2 - h - y
                else:
                    dy = y2 + h2 - y
                if abs(dx) < abs(dy):
                    x += dx
                else:
                    y += dy
        # print(furniture['name'], x, y, w, h)
        # STEP 2: If the furniture is out of the room, move it inside
        if x < 12:
            x = 12
        if y < 12:
            y = 12
        if x+w > img.shape[1]-12:
            x = img.shape[1]-12-w
        if y+h > img.shape[0]-12:
            y = img.shape[0]-12-h
        furniture_new['x'] = x
        furniture_new['y'] = y

        furnitures_new.append(furniture_new)
    data['furniture'] = furnitures_new
    furnitures = data['furniture']
    furnitures_new = []
    for furniture in furnitures:
        furniture_new = furniture.copy()
        x = furniture['x']
        y = furniture['y']
        w = furniture['w']
        h = furniture['h']
        # STEP 3: Snap to the nearest wall
        max = x
        direction = ''
        name = furniture_new['name']
        if name not in ['Sofa','Dining Chair', 'Coffee Table', 'Dining Table']:
            for x_wall in xs:
                if x > x_wall and x - x_wall < max:
                    max = x - x_wall
                    direction = 'left'
                if x + w < x_wall and x_wall - (x + w) < max:
                    max = x_wall - (x + w)
                    direction = 'right'
            for y_wall in ys:
                if y > y_wall and y - y_wall < max:
                    max = y - y_wall
                    direction = 'up'
                if y + h < y_wall and y_wall - (y + h) < max:
                    max = y_wall - (y + h)
                    direction = 'down'
            if direction == 'left':
                x -= max
            if direction == 'right':
                x += max
            if direction == 'up':
                y -= max
            if direction == 'down':
                y += max
            furniture_new['x'] = x
            furniture_new['y'] = y
            furnitures_new.append(furniture_new)
        name = furniture_new['name']
        if name in ['Standing Lamp', 'Table Lamp']:
            # draw the lamp use large circle, small circle and cross
            cv2.circle(img, (x+w//2, y+h//2), w//2, (0,0,0), 1)
            cv2.circle(img, (x+w//2, y+h//2), w//4, (0,0,0), 1)
            cv2.line(img, (x+w//2-w//4, y+h//2), (x+w//2+w//4, y+h//2), (0,0,0), 1)
            cv2.line(img, (x+w//2, y+h//2-w//4), (x+w//2, y+h//2+w//4), (0,0,0), 1)
        elif name in ['Bed', 'Bed Head', 'Bed Platform', 'Bed Storage']:
            # draw the bed use large rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
            if w >= h:
                cv2.line(img, (x+w//3, y), (x+w//2, y+h), (0,0,0), 1)
                cv2.line(img, (x+w//2, y), (x+w//2, y+h), (0,0,0), 1)
                # draw the bed head use small rectangle
                cv2.rectangle(img, (x, y), (x+w//12, y+h), (0,0,0), 1)
                if w <= 2*h:
                    #draw 2 pillows
                    cv2.rectangle(img, (x+w//6, y+h//12), (x+w//3, y+h//2), (0,0,0), 1)
                    cv2.rectangle(img, (x+w//6, y+h//2), (x+w//3, y+h-h//12), (0,0,0), 1)
                else:
                    #draw 1 pillow
                    cv2.rectangle(img, (x+w//6, y+h//12), (x+w//3, y+h-h//12), (0,0,0), 1)
            else:
                cv2.line(img, (x, y+h//3), (x+w, y+h//2), (0,0,0), 1)
                cv2.line(img, (x, y+h//2), (x+w, y+h//2), (0,0,0), 1)
                # draw the bed head use small rectangle
                cv2.rectangle(img, (x, y), (x+w, y+h//12), (0,0,0), 1)
                if h <= 2*w:
                    #draw 2 pillows
                    cv2.rectangle(img, (x+w//12, y+h//6), (x+w//2, y+h//3), (0,0,0), 1)
                    cv2.rectangle(img, (x+w//2, y+h//6), (x+w-w//12, y+h//3), (0,0,0), 1)
                else:
                    #draw 1 pillow
                    cv2.rectangle(img, (x+w//12, y+h//6), (x+w-w//12, y+h//3), (0,0,0), 1)
            
        elif name in ['Bedside Table', 'Coffee Table', 'Dining Table', 'Dressing Table', 'Non Paid Study Table', 'Study Table']:
            # draw the table use large rectangle with 4 legs
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
            cv2.rectangle(img, (x, y), (x+2, y+2), (0,0,0), 1)
            cv2.rectangle(img, (x+w-2, y), (x+w, y+2), (0,0,0), 1)
            cv2.rectangle(img, (x, y+h-2), (x+2, y+h), (0,0,0), 1)
            cv2.rectangle(img, (x+w-2, y+h-2), (x+w, y+h), (0,0,0), 1)
        elif name in ['Shoe Cabinet', 'Storage Cabinets', 'Tall Cabinet', 'Wardrobes','Bottom Island Kitchen Cabinet', 'Bottom Kitchen Cabinet', 'Bottom Vanity Cabinet','Top Kitchen Cabinet', 'Top Vanity Cabinet Storage']:
            # draw the cabinet use large rectangle with a cross inside
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
            cv2.line(img, (x, y), (x+w, y+h), (0,0,0), 1)
            cv2.line(img, (x, y+h), (x+w, y), (0,0,0), 1)
        elif name in ['Toilet','Toilet Bowl','Basin', 'Toilet Sink']:
            # draw the toilet use large ellipse with a small ellipse inside
            cv2.ellipse(img, (x+w//2, y+h//2), (w//2, h//2), 0, 0, 360, (0,0,0), 1)
            cv2.ellipse(img, (x+w//2, y+h//2), (w//3, h//3), 0, 0, 360, (0,0,0), 1)
        elif name in ['Dining Chair']:
            # draw the chair use large rectangle with 4 legs and back
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
            cv2.rectangle(img, (x, y), (x+1, y+1), (0,0,0), 1)
            cv2.rectangle(img, (x+w-1, y), (x+w, y+1), (0,0,0), 1)
            cv2.rectangle(img, (x, y+h-1), (x+1, y+h), (0,0,0), 1)
            cv2.rectangle(img, (x+w-1, y+h-1), (x+w, y+h), (0,0,0), 1)
        elif name in ['Sofa']:
            # draw the sofa use large rectangle with a small rectangle inside and 2 arms
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
            cv2.rectangle(img, (x+w//4, y+h//4), (x+w-w//4, y+h-h//4), (0,0,0), 1)
            cv2.rectangle(img, (x, y+h//4), (x+w//4, y+h-h//4), (0,0,0), 1)
            cv2.rectangle(img, (x+w-w//4, y+h//4), (x+w, y+h-h//4), (0,0,0), 1)
        else:
            # draw the furniture use large rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,0), 1)
        furniture_new['x'] = int(x + w//2)
        furniture_new['y'] = int(y + h//2)
    data['furniture'] = furnitures_new
    json.dump(data, open(filepath, 'w'))
    file_name = os.path.basename(filepath).split('.')[0] + '.png'
    output_path = os.path.join(imgpath, file_name)
    cv2.imwrite(output_path, img)