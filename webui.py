import shutil

import gradio as gr
import os
import sys
import subprocess as sub
import glob
import json
import torch
import multiprocessing as mp
from pathlib import Path

def verbose_print(text):
    if "verbose" in model_config_df and model_config_df["verbose"]:
        print(f"{text}")

def update_merged_dirs():
    temp_merge_dirs = []
    if ("dataset_path" in dataset_config_df) and (not dataset_config_df["dataset_path"] == ''):
        all_files = os.listdir(dataset_config_df["dataset_path"])
        for path in all_files:
            path = os.path.join(str(dataset_config_df["dataset_path"]), path)
            if os.path.isdir(path):
                verbose_print(f'sub-directory:\t{path}')
                temp_merge_dirs.append(path)
    return temp_merge_dirs

# set local path
cwd = os.getcwd()

# get list of all .ckpt files in directory & set model option buttons
ckpt_files = glob.glob(os.path.join(cwd, '*.ckpt'))
ckpt_files = [f.split('/')[-1] for f in ckpt_files]

# create save file for everything to be written to (open/overwrite/close) whenever changes are saved
model_config_df = {}
dataset_config_df = {}
image_gen_config_df = {}
train_config_df = {}
system_config_df = {}

json_file_name = "gui_params.json"
file_exists = os.path.exists(json_file_name)
if not file_exists:
    with open(json_file_name, 'w') as f:
        f.close()
else:
    data_flag = True
    with open(json_file_name, 'r') as json_file:
        lines = json_file.readlines()
        if len(lines) == 0 or len(lines[0].replace(' ', '')) == 0:
            data_flag = False
        json_file.close()

    if data_flag:
        with open(json_file_name) as json_file:
            data = json.load(json_file)

            model_config_df = [dictionary for dictionary in data if "model_name" in dictionary]
            if len(model_config_df) > 0:
                model_config_df = model_config_df[0]
            else:
                model_config_df = {}
            dataset_config_df = [dictionary for dictionary in data if "config_path" in dictionary]
            if len(dataset_config_df) > 0:
                dataset_config_df = dataset_config_df[0]
            else:
                dataset_config_df = {}
            system_config_df = [dictionary for dictionary in data if "gpu_used_var" in dictionary]
            if len(system_config_df) > 0:
                system_config_df = system_config_df[0]
            else:
                system_config_df = {}
            image_gen_config_df = [dictionary for dictionary in data if "ddim_eta_var" in dictionary]
            if len(image_gen_config_df) > 0:
                image_gen_config_df = image_gen_config_df[0]
            else:
                image_gen_config_df = {}
            train_config_df = [dictionary for dictionary in data if "max_training_steps" in dictionary]
            if len(train_config_df) > 0:
                train_config_df = train_config_df[0]
            else:
                train_config_df = {}
            del data
            json_file.close()

dataset_merge_dirs = update_merged_dirs()
verbose_print(dataset_merge_dirs)

def execute(cmd):
    popen = sub.Popen(cmd, stdout=sub.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise sub.CalledProcessError(return_code, cmd)

def dependency_install_button():
    return sub.run(f"pip install -r {cwd}/requirements.txt".split(" "), stdout=sub.PIPE).stdout.decode("utf-8")

def update_JSON():
    temp = [model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df]
    for entry in temp:
        verbose_print(entry)

    with open(json_file_name, "w") as f:
        json.dump([model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df], indent=4, fp=f)
    f.close()
    verbose_print("="*42)

def create_data_dirs():
    dataset_type = ["dataset_path", "reg_dataset_path"]
    for i in range(0, len(dataset_type)):
        if not os.path.exists(dataset_config_df[dataset_type[i]]) and not dataset_config_df[dataset_type[i]] == '':
            dataset = dataset_config_df[dataset_type[i]]
            dir_create_str = f"mkdir -p {dataset}"
            sub.run(dir_create_str.split(" "))

def model_choice(ver):
    for i in range(0, len(ckpt_files)):
        if ver == ckpt_files[i]:
            return ver
    return None

def verbose_checkbox():
    model_config_df["verbose"] = not model_config_df["verbose"]

def model_config_save_button(model_name, gpu_used_var, project_name, class_token, config_path, dataset_path, reg_dataset_path):
    model_config_df["model_name"] = model_name
    system_config_df["gpu_used_var"] = int(gpu_used_var.replace("gpu: ", ""))
    dataset_config_df["project_name"] = project_name
    dataset_config_df["class_token"] = class_token
    dataset_config_df["config_path"] = config_path
    dataset_config_df["dataset_path"] = dataset_path
    dataset_config_df["reg_dataset_path"] = reg_dataset_path

    image_gen_config_df["final_img_path"] = dataset_config_df["reg_dataset_path"]

    all_lines = None
    is_target_same = False
    if not dataset_config_df["class_token"] == '':
        with open(os.path.join(cwd, "ldm/data/personalized.py"), "r") as script_file:
            lines = script_file.readlines()
            for i in range(0, len(lines)):
                if "{}" in lines[i]:
                    prior_class = (lines[i].split("\'")[1]).split(" ")[0]
                    if prior_class == dataset_config_df["class_token"]:
                        is_target_same = True
                        break
                    lines[i] = lines[i].replace(prior_class, str(dataset_config_df["class_token"]))
                    all_lines = lines
                    break
            script_file.close()
        if not is_target_same:
            with open(os.path.join(cwd, "ldm/data/personalized.py"), "w") as script_file:
                for line in all_lines:
                    script_file.write(line)
                script_file.close()

    # update json file
    update_JSON()

    # create directories if necessary
    create_data_dirs()

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'subdirs:\t{sub_dir_names}')

    return reg_dataset_path, gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", interactive=True, value=[False for name in sub_dir_names])

def change_regularizer_view(choice):
    if "Custom" in choice:
        image_gen_config_df["regularizer_var"] = 0
    elif "Auto" in choice:
        image_gen_config_df["regularizer_var"] = 1

def image_gen_config_save_button(final_img_path, seed_var, ddim_eta_var, scale_var, prompt_string, n_samples, n_iter, ddim_steps, keep_jpgs):
    image_gen_config_df["final_img_path"] = final_img_path
    image_gen_config_df["seed_var"] = int(seed_var)
    image_gen_config_df["ddim_eta_var"] = float(ddim_eta_var)
    image_gen_config_df["scale_var"] = float(scale_var)
    image_gen_config_df["prompt_string"] = prompt_string
    image_gen_config_df["n_samples"] = int(n_samples)
    image_gen_config_df["n_iter"] = int(n_iter)
    image_gen_config_df["ddim_steps"] = int(ddim_steps)
    image_gen_config_df["keep_jpgs"] = bool(keep_jpgs)

    dataset_config_df["reg_dataset_path"] = image_gen_config_df["final_img_path"]

    # update json file
    update_JSON()

    # create directories if necessary
    create_data_dirs()

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'subdirs:\t{sub_dir_names}')

    return final_img_path, gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", interactive=True, value=[False for name in sub_dir_names])

def image_generation_button(keep_jpgs):
    prompt = image_gen_config_df['prompt_string'].replace('_', ' ')
    image_gen_cmd = f"python scripts/stable_txt2img.py --seed {image_gen_config_df['seed_var']} " \
                    f"--ddim_eta {image_gen_config_df['ddim_eta_var']} --n_samples {image_gen_config_df['n_samples']} " \
                    f"--n_iter {image_gen_config_df['n_iter']} --scale {image_gen_config_df['scale_var']} " \
                    f"--ddim_steps {image_gen_config_df['ddim_steps']} --ckpt {model_config_df['model_name']} " \
                    f"--prompt \'{prompt}\' --outdir {image_gen_config_df['final_img_path']}"

    if keep_jpgs:
        image_gen_cmd = f"{image_gen_cmd} --keep_jpgs"

    verbose_print("============================== IMAGE GENERATION TEST ==============================")
    verbose_print(image_gen_cmd)
    verbose_print("============================== --------------------- ==============================")

    if ("regularizer_var" in image_gen_config_df and image_gen_config_df["regularizer_var"] == 1):
        if "seed_var" in image_gen_config_df and "ddim_eta_var" in image_gen_config_df and \
                "n_samples" in image_gen_config_df and "n_iter" in image_gen_config_df and \
                "scale_var" in image_gen_config_df and "ddim_steps" in image_gen_config_df and \
                "model_name" in model_config_df and "prompt_string" in image_gen_config_df and "final_img_path" in image_gen_config_df:
            for line in execute(image_gen_cmd.split(" ")):
                verbose_print(line)
    else:
        if "seed_var" in image_gen_config_df and "ddim_eta_var" in image_gen_config_df and \
                "n_samples" in image_gen_config_df and "n_iter" in image_gen_config_df and \
                "scale_var" in image_gen_config_df and "ddim_steps" in image_gen_config_df and \
                "model_name" in model_config_df and "prompt_string" in image_gen_config_df and "final_img_path" in image_gen_config_df:
            for line in execute(image_gen_cmd.split(" ")):
                verbose_print(line)

def train_save_button(max_training_steps, batch_size, cpu_workers, model_path):
    train_config_df['max_training_steps'] = int(max_training_steps)
    train_config_df['batch_size'] = int(batch_size)
    train_config_df['cpu_workers'] = int(cpu_workers)
    train_config_df['model_path'] = model_path if (model_path and not model_path == '') else None if (not 'model_path' in image_gen_config_df) else (image_gen_config_df['model_path'])

    # update json file
    update_JSON()

    # create directories if necessary
    create_data_dirs()

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'subdirs:\t{sub_dir_names}')

    return gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", interactive=True, value=[False for name in sub_dir_names])

def train_resume_checkbox(checkbox):
    if checkbox:
        if 'model_path' in train_config_df:
            return gr.update(label='Path to ckpt model in logs directory', value=str(train_config_df['model_path']), visible=True, interactive=True)
        else:
            return gr.update(label='Path to ckpt model in logs directory', visible=True, interactive=True)
    else:
        return gr.update(visible=False)
def prune_ckpt():
    temp_path = os.path.join(cwd, 'logs')
    paths = sorted(Path(temp_path).iterdir(), key=os.path.getmtime)
    verbose_print(paths)
    temp_path = os.path.join(temp_path, paths[-1])
    temp_path = os.path.join(temp_path, 'checkpoints/last.ckpt')
    verbose_print(temp_path)

    train_config_df["model_path"] = temp_path

    prune_cmd = f"python prune-ckpt.py --ckpt {temp_path}"
    execute(prune_cmd)
    verbose_print(f"Model Pruning Complete!")

def train_button(train_resume_var):
    # train the model
    prompt = image_gen_config_df['prompt_string'].replace('_', ' ')
    train_cmd = f"python main.py --base {dataset_config_df['config_path']} -t --actual_resume {model_config_df['model_name']} " \
                f"--reg_data_root {image_gen_config_df['final_img_path']} -n {dataset_config_df['project_name']} " \
                f"--gpus {system_config_df['gpu_used_var']}, --data_root {dataset_config_df['dataset_path']} " \
                f"--max_training_steps {train_config_df['max_training_steps']} --class_word {prompt} --token {dataset_config_df['class_token']} " \
                f"--no-test --batch_size {train_config_df['batch_size']} --workers {train_config_df['cpu_workers']}"

    if train_resume_var:
        train_cmd = f"{train_cmd} --resume --actual_resume {train_config_df['model_path']}"

    verbose_print("============================== TRAINING COMMAND TEST ==============================")
    verbose_print(train_cmd)
    verbose_print("============================== --------------------- ==============================")

    if ("regularizer_var" in image_gen_config_df and image_gen_config_df["regularizer_var"] == 1):
        if 'config_path' in dataset_config_df and 'model_name' in model_config_df and \
                'final_img_path' in image_gen_config_df and 'project_name' in dataset_config_df and \
                'gpu_used_var' in system_config_df and 'dataset_path' in dataset_config_df and \
                'max_training_steps' in train_config_df and prompt:
            for line in execute(train_cmd.split(" ")):
                verbose_print(line)
    else:
        if 'config_path' in dataset_config_df and 'model_name' in model_config_df and \
                'final_img_path' in image_gen_config_df and 'project_name' in dataset_config_df and \
                'gpu_used_var' in system_config_df and 'dataset_path' in dataset_config_df and \
                'max_training_steps' in train_config_df and prompt:
            for line in execute(train_cmd.split(" ")):
                verbose_print(line)

    return gr.update(value="Prune Model", variant='secondary', visible=True)


def merge_data_button(merge_data_list_var):
    global dataset_merge_dirs

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'sub_dir_names:\t{sub_dir_names}')

    # remove the booleans
    merge_data_list_var = list(filter(lambda x:not(type(x) is bool), merge_data_list_var))

    temp_list = [False]*len(dataset_merge_dirs)
    for element in merge_data_list_var:
        temp_list[sub_dir_names.index(element)] = True
    merge_data_list_var = temp_list

    paths_to_merge = []
    for i in range(0, len(merge_data_list_var)):
        if merge_data_list_var[i]:
            paths_to_merge.append(dataset_merge_dirs[i])

    verbose_print(f'directories to merge:\t{paths_to_merge}')

    # loop for all sub-folders
    for sub_dir_path in paths_to_merge:
        ### count total in base dataset_path
        total = len(glob.glob1(str(dataset_config_df["dataset_path"]), "*.png"))
        ### get images from sub-dir
        images_list = glob.glob1(sub_dir_path, "*.png")
        ### move images from each sub-dir into the dataset_path
        counter = total
        for image in images_list:
            original_image_path = os.path.join(sub_dir_path, image)
            os.rename(original_image_path, os.path.join(str(dataset_config_df["dataset_path"]), f"{counter}.png"))
            counter += 1
        verbose_print(f'Done Merging {sub_dir_path} into {dataset_config_df["dataset_path"]}')
        ### delete redundant directory
        shutil.rmtree(sub_dir_path, ignore_errors=True)

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    return gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", interactive=True, value=[False for name in sub_dir_names])

with gr.Blocks() as demo:
    with gr.Tab("Model & Data Configuration"):
        config_save_var = gr.Button(value="Apply & Save Settings", variant='primary')
        gr.Markdown(
        """
        ### Make sure a stable diffusion model with the (.ckpt) extension has been downloaded
        ### Please move the downloaded model into "this" repository folder
        """)

        verbose_print(f"ckpt_files {ckpt_files}")

        with gr.Row():
            if "model_name" in model_config_df:
                model_var = gr.inputs.Radio(ckpt_files, type="value", default=str(model_config_df["model_name"]), label='Select Model')
            else:
                model_var = gr.inputs.Radio(ckpt_files, type="value", label='Select Model')

        if "project_name" in dataset_config_df:
            project_name = gr.Textbox(lines=1, interactive=True, label='Project Name', value=str(dataset_config_df["project_name"]))
        else:
            project_name = gr.Textbox(lines=1, interactive=True, label='Project Name')
        if "class_token" in dataset_config_df:
            class_token = gr.Textbox(lines=1, interactive=True, label='Token (e.g. firstnamelastname)', value=str(dataset_config_df["class_token"]))
        else:
            class_token = gr.Textbox(lines=1, interactive=True, label='Token (e.g. firstnamelastname)')
        if "config_path" in dataset_config_df:
            config_path = gr.Textbox(lines=1, interactive=True, label='Path to Model YAML Config', value=str(dataset_config_df["config_path"]))
        else:
            config_path = gr.Textbox(lines=1, interactive=True, label='Path to Model YAML Config')
        if "dataset_path" in dataset_config_df:
            dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Class Target Dataset', value=str(dataset_config_df["dataset_path"]))
        else:
            dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Class Target Dataset')
        if "reg_dataset_path" in dataset_config_df:
            reg_dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Regularization Dataset', value=str(dataset_config_df["reg_dataset_path"]))
        else:
            reg_dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Regularization Dataset')

        with gr.Row():
            with gr.Column():
                if "verbose" in model_config_df:
                    verbose = gr.Checkbox(interactive=True, label='Verbose Mode', value=bool(model_config_df["verbose"]))
                else:
                    model_config_df["verbose"] = False
                    verbose = gr.Checkbox(interactive=True, label='Verbose Mode', value=False)
            with gr.Column():
                if not "gpu_used_var" in system_config_df:
                    system_config_df["gpu_used_var"] = [i for i in range(0, torch.cuda.device_count())][0] # EXPECT THIS TO CHANGE IN THE FUTURE
                temp_text = [f"gpu: {i}" for i in range(0, torch.cuda.device_count())]
                gpu_used_var = gr.inputs.Radio(temp_text, default=temp_text[(system_config_df["gpu_used_var"])], label='Select GPU')

    with gr.Tab("Image Regularization"):
        gr.Markdown(
        """
        ### Please make sure that if you are using a custom regularization dataset, that the images are formatted 512x512
        ### Here is an online formatter for cropping images if needed [https://www.birme.net/](https://www.birme.net/)
        """)
        regularizer_save_var = gr.Button(value="Apply & Save Settings", variant='primary')
        image_gen_output = None
        with gr.Row():
            reg_options = ['Custom (REG) Images (🔺 stylization, 🔻 robustness)', 'Auto (REG) Images (🔺 robustness,🔻 stylization)']
            if "regularizer_var" in image_gen_config_df:
                regularizer_var = gr.inputs.Radio(reg_options,
                                type="value", default=reg_options[(image_gen_config_df["regularizer_var"])], label='Select Regularization Approach')
            else:
                regularizer_var = gr.inputs.Radio(reg_options,
                                type="value", label='Select Regularization Approach')
            if not "final_img_path" in image_gen_config_df and not "reg_dataset_path" in dataset_config_df:
                final_img_path = gr.Textbox(lines=1, interactive=True, label='Regularization Data Path')
            elif "reg_dataset_path" in dataset_config_df:
                final_img_path = gr.Textbox(lines=1, interactive=True, label='Regularization Data Path',
                                            value=str(dataset_config_df["reg_dataset_path"]))
            elif "final_img_path" in image_gen_config_df:
                final_img_path = gr.Textbox(lines=1, interactive=True, label='Regularization Data Path',
                                            value=str(image_gen_config_df["final_img_path"]))

        if "seed_var" in image_gen_config_df:
            seed_var = gr.Textbox(lines=1, interactive=True, label='Seed Number (int)', value=str(image_gen_config_df["seed_var"]))
        else:
            seed_var = gr.Textbox(lines=1, interactive=True, label='Seed Number (int)', value=str(10))
        if "ddim_eta_var" in image_gen_config_df:
            ddim_eta_var = gr.Textbox(lines=1, interactive=True, label='DDIM eta (float)', value=str(image_gen_config_df["ddim_eta_var"]))
        else:
            ddim_eta_var = gr.Textbox(lines=1, interactive=True, label='DDIM eta (float)', value=str(0.0))
        if "scale_var" in image_gen_config_df:
            scale_var = gr.Textbox(lines=1, interactive=True, label='Scale (float)', value=str(image_gen_config_df["scale_var"]))
        else:
            scale_var = gr.Textbox(lines=1, interactive=True, label='Scale (float)', value=str(10.0))

        with gr.Row():
            with gr.Column():
                if "prompt_string" in image_gen_config_df:
                    prompt_string = gr.Textbox(lines=1, interactive=True, label='Regularization Prompt (e.g. Person)', value=str(image_gen_config_df["prompt_string"]))
                else:
                    prompt_string = gr.Textbox(lines=1, interactive=True, label='Regularization Prompt (e.g. Person)')
            with gr.Column():
                if "keep_jpgs" in image_gen_config_df:
                    keep_jpgs = gr.Checkbox(interactive=True, label='Keep JPGs', value=bool(image_gen_config_df["keep_jpgs"]))
                else:
                    image_gen_config_df["keep_jpgs"] = False
                    keep_jpgs = gr.Checkbox(interactive=True, label='Keep JPGs', value=False)

        if "n_samples" in image_gen_config_df:
            n_samples = gr.Slider(minimum=0, maximum=200, step=1, label='Samples per Iteration (i.e. batch size)', value=int(image_gen_config_df["n_samples"]))
        else:
            n_samples = gr.Slider(minimum=0, maximum=200, step=1, value=1, label='Samples per Iteration (i.e. batch size)')
        if "n_iter" in image_gen_config_df:
            n_iter = gr.Slider(minimum=0, maximum=1000, step=10, label='Number of Iterations', value=int(image_gen_config_df["n_iter"]))
        else:
            n_iter = gr.Slider(minimum=0, maximum=1000, step=10, value=200, label='Number of Iterations')
        if "ddim_steps" in image_gen_config_df:
            ddim_steps = gr.Slider(minimum=0, maximum=250, step=5, label='Sampler Steps', value=int(image_gen_config_df["ddim_steps"]))
        else:
            ddim_steps = gr.Slider(minimum=0, maximum=250, step=5, value=50, label='Sampler Steps')

        with gr.Row():
            generate_images_var = gr.Button(value="Generate", variant='secondary')

    with gr.Tab("Fine-Tuning Model"):
        fine_tine_save_var = gr.Button(value="Apply & Save Settings", variant='primary')

        if "max_training_steps" in train_config_df:
            max_training_steps = gr.Slider(minimum=0, maximum=20000, step=20, label='Max Training Steps', value=int(train_config_df["max_training_steps"]))
        else:
            max_training_steps = gr.Slider(minimum=0, maximum=20000, step=20, label='Max Training Steps', value=2000)
        if "batch_size" in train_config_df:
            batch_size = gr.Slider(minimum=1, maximum=64, step=1, label='Batch Size', value=int(train_config_df["batch_size"]))
        else:
            batch_size = gr.Slider(minimum=1, maximum=64, step=1, label='Batch Size', value=1)
        if "cpu_workers" in train_config_df:
            cpu_workers = gr.Slider(minimum=1, maximum=mp.cpu_count(), step=1, label='Worker Threads', value=int(train_config_df["cpu_workers"]))
        else:
            cpu_workers = gr.Slider(minimum=1, maximum=mp.cpu_count(), step=1, label='Worker Threads', value=int(mp.cpu_count()/2))

        with gr.Row():
            train_out_var = gr.Button(value="Train", variant='secondary')
            prune_model_var = gr.Button(visible=False)

        with gr.Accordion("Trying to Resume Training? OR Merge Data Directories? (Look Here!)"):
            gr.Markdown(
                """
                ### Make sure a you have the full checkpoint directory in the ( logs ) directory :: if resuming training
                ### Make sure to check all sub-directories you want merged within the path of your data directory
                ### ( IMPORTANT ) if merging, be aware that the data merged is (not copied); please back up sub-directories if you want to keep their contents!
                ### ( EVEN MORE IMPORTANT ) make sure that if merging sub-directories, that there is no left over data still in the current dataset_path. In other words make sure images are in some kind of sub-directory. (otherwise data could be overwritten!)
                ### To refresh the list of sub-directories (checkboxes) after merging, then go to the first tab in the UI and click ( APPLY SETTINGS )
                """)
            with gr.Row():
                train_resume_var = gr.Checkbox(interactive=True, label='Resume Training', value=False)
                model_path_var = gr.Textbox(visible=False)
            with gr.Row():
                with gr.Row():
                    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
                    merge_data_list_var = gr.CheckboxGroup(choices=sub_dir_names, label="Dataset Sub-Directories", interactive=True, value=[False for name in sub_dir_names])
                with gr.Row():
                    merge_data_button_var = gr.Button(value="Merge All Data Sub-Directories", variant='secondary')

    model_var.change(fn=model_choice, inputs=[model_var], outputs=[])
    config_save_var.click(fn=model_config_save_button,
                          inputs=[model_var,
                                  gpu_used_var,
                                  project_name,
                                  class_token,
                                  config_path,
                                  dataset_path,
                                  reg_dataset_path
                                  ],
                          outputs=[final_img_path, merge_data_list_var]
                          )
    verbose.change(fn=verbose_checkbox, inputs=[], outputs=[])

    regularizer_var.change(fn=change_regularizer_view, inputs=[regularizer_var], outputs=[])
    regularizer_save_var.click(fn=image_gen_config_save_button, inputs=[final_img_path,
                                                                        seed_var,
                                                                        ddim_eta_var,
                                                                        scale_var,
                                                                        prompt_string,
                                                                        n_samples,
                                                                        n_iter,
                                                                        ddim_steps,
                                                                        keep_jpgs
                                                                        ], outputs=[reg_dataset_path, merge_data_list_var])

    generate_images_var.click(fn=image_generation_button, inputs=[keep_jpgs], outputs=[], show_progress=True, scroll_to_output=True)

    fine_tine_save_var.click(fn=train_save_button, inputs=[max_training_steps,
                                                                        batch_size,
                                                                        cpu_workers,
                                                                        model_path_var
                                                                        ], outputs=[merge_data_list_var])
    train_out_var.click(fn=train_button, inputs=[train_resume_var], outputs=[prune_model_var], show_progress=True, scroll_to_output=True)
    prune_model_var.click(fn=prune_ckpt, inputs=[], outputs=[])
    train_resume_var.change(fn=train_resume_checkbox, inputs=[train_resume_var], outputs=[model_path_var])

    merge_data_button_var.click(fn=merge_data_button, inputs=[merge_data_list_var], outputs=[merge_data_list_var])

if __name__ == "__main__":
    demo.launch()

