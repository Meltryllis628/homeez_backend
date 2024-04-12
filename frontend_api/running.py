from .parsing import parse_input_json,generate_furniture
from .evaluating import generate_outputs
from .draw import draw_room
import os
import zipfile

def run_generation(path):
    print("Parsing input...",end='')
    input_path = parse_input_json(path)
    root_input_file_name = os.path.basename(input_path)
    raw_jsons_dir = os.path.join(input_path,'json_raw')
    json_dir = os.path.join(input_path,'jsons')
    img_dir = os.path.join(input_path,'img')
    if not os.path.exists(json_dir):
        os.makedirs(json_dir)
    for file in os.listdir(raw_jsons_dir):
        if file.endswith('.json'):
            generate_furniture(os.path.join(raw_jsons_dir,file),json_dir,img_dir)
            file_name = file.split('.')[0]
    input_jsons = [os.path.join(json_dir,file) for file in os.listdir(json_dir) if file.endswith('.json')]
    image_dest_directory = os.path.join(input_path,'img_out')
    output_directory = os.path.join(input_path,'json_out')
    print("Done")
    print("Generating images...",end='')
    generate_outputs(input_jsons, img_dir, image_dest_directory,  output_directory)
    print("Done")
    print("Drawing rooms...",end='')
    output_jsons = [os.path.join(output_directory,file) for file in os.listdir(output_directory) if file.endswith('.json')]
    for file in output_jsons:
        draw_room(file, image_dest_directory)
    print("Done")
    print("Zipping files...",end='')
    output_imgs = [os.path.join(image_dest_directory,file) for file in os.listdir(image_dest_directory) if file.endswith('.png')]
    zip_files = output_jsons + output_imgs
    with zipfile.ZipFile(os.path.join(input_path,f'{root_input_file_name}.zip'),'w') as zipf:
        for file in zip_files:
            zipf.write(file,os.path.basename(file))
    print("Done")
    return os.path.join(input_path,f'{root_input_file_name}.zip')