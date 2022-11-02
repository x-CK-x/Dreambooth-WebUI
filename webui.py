import math
import shutil
from PIL import Image
import gradio as gr
import os
import sys
import subprocess as sub
import glob
import json
import torch
import multiprocessing as mp
from pathlib import Path
import copy

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

# session config
session_file_name = "gui_params.json"

# presets config
presets_file_name = "presets_params.json"

model_config_df = {}
dataset_config_df = {}
system_config_df = {}
image_gen_config_df = {}
train_config_df = {}

def load_session_config(f_name):
    global model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df
    file_exists = os.path.exists(f_name)
    if not file_exists:
        with open(f_name, 'w') as f:
            f.close()
    else:
        data_flag = True # detects if the file is empty
        with open(f_name, 'r') as json_file:
            lines = json_file.readlines()
            if len(lines) == 0 or len(lines[0].replace(' ', '')) == 0:
                data_flag = False
            json_file.close()

        if data_flag:
            with open(f_name) as json_file:
                data = json.load(json_file)
                temp_config = [dictionary for dictionary in data if "model_name" in dictionary]
                if len(temp_config) > 0:
                    model_config_df = temp_config[0]
                else:
                    model_config_df = {}

                temp_config = [dictionary for dictionary in data if "config_path" in dictionary]
                if len(temp_config) > 0:
                    dataset_config_df = temp_config[0]
                else:
                    dataset_config_df = {}

                temp_config = [dictionary for dictionary in data if "gpu_used_var" in dictionary]
                if len(temp_config) > 0:
                    system_config_df = temp_config[0]
                else:
                    system_config_df = {}

                temp_config = [dictionary for dictionary in data if "ddim_eta_var" in dictionary]
                if len(temp_config) > 0:
                    image_gen_config_df = temp_config[0]
                else:
                    image_gen_config_df = {}

                temp_config = [dictionary for dictionary in data if "max_training_steps" in dictionary]
                if len(temp_config) > 0:
                    train_config_df = temp_config[0]
                else:
                    train_config_df = {}
                json_file.close()

def load_presets_config(f_name):
    global presets
    presets = []

    file_exists = os.path.exists(f_name)
    if not file_exists:
        with open(f_name, 'w') as f:
            f.close()
    else:
        data_flag = True  # detects if the file is empty
        with open(f_name, 'r') as json_file:
            lines = json_file.readlines()
            if len(lines) == 0 or len(lines[0].replace(' ', '')) == 0:
                data_flag = False
            json_file.close()

        if data_flag:
            with open(f_name) as json_file:
                data = json.load(json_file)
                presets = data # is already expected to be a list
                json_file.close()
    return presets

# create save file for everything to be written to (open/overwrite/close) whenever changes are saved
# load session
load_session_config(session_file_name)

dataset_merge_dirs = update_merged_dirs()
verbose_print(dataset_merge_dirs)

# load presets
presets = load_presets_config(presets_file_name)
verbose_print(f"presets_file_name:::::::\t{presets}") # A LIST OF DICTIONARIES!

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

def update_JSON(is_preset):
    global model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df

    if is_preset:
        for entry in presets:
            verbose_print(entry)

        with open(presets_file_name, "w") as f:
            json.dump(presets, indent=4, fp=f) # assume it's a list already
        f.close()
        verbose_print("@"*42)
    else:
        temp = copy.deepcopy([model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df])
        for entry in temp:
            verbose_print(entry)

        with open(session_file_name, "w") as f:
            json.dump(temp, indent=4, fp=f)
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

def model_config_save_button(model_name, gpu_used_var, project_name, class_token, config_path, dataset_path, reg_dataset_path, preset_name_var):
    global model_config_df, dataset_config_df, system_config_df
    verbose_print(f"==========----- MODEL CONFIG SAVE -----==========")
    verbose_print(f"BEFORE:::\tpresets:\t{presets}")

    model_config_df["model_name"] = model_name
    if gpu_used_var and isinstance(gpu_used_var, str) and len(gpu_used_var) > 0:
        system_config_df["gpu_used_var"] = int(gpu_used_var.replace("gpu: ", ""))
    dataset_config_df["project_name"] = project_name
    dataset_config_df["class_token"] = class_token
    dataset_config_df["config_path"] = config_path
    dataset_config_df["dataset_path"] = dataset_path
    dataset_config_df["reg_dataset_path"] = reg_dataset_path

    image_gen_config_df["final_img_path"] = dataset_config_df["reg_dataset_path"]

    model_config_df["preset_name"] = preset_name_var
    verbose_print(f"MIDDLE:::\tpresets:\t{presets}")

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
    update_JSON(is_preset=False)

    # create directories if necessary
    create_data_dirs()

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'subdirs:\t{sub_dir_names}')
    verbose_print(f"AFTER:::\tpresets:\t{presets}")
    verbose_print(f"==========----- MODEL CONFIG SAVE -----========== done")
    return reg_dataset_path, gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])

def get_all_presets_keys():
    global presets
    verbose_print(f"total amount of presets:\t{len(presets)}")
    temp = [list(preset.keys()) for preset in presets]
    temp = [item for sublist in temp for item in sublist]

    verbose_print(f"key list:\t{temp}")

    return temp
def get_preset_dict(name):
    global presets
    for preset in presets:
        for key in list(preset.keys()):
            if key == name:
                return preset
    verbose_print("NO PRESET DETECTED")
    raise ValueError(f"NO PRESET DETECTED! Either this was an INVALID name, or does the preset not exist?\tPresets Available:\t{presets}\nName:\t{name}")

def get_preset_index(name):
    global presets
    for i in range(0, len(presets)):
        for key in list(presets[i].keys()):
            if key == name:
                return i
    verbose_print("NO PRESET DETECTED")
    raise ValueError(f"NO PRESET DETECTED! Either this was an INVALID name, or does the preset not exist?\tPresets Available:\t{presets}\nName:\t{name}")

def remove_preset_from_list(key):
    global presets
    index_to_remove = get_preset_index(key)
    del presets[index_to_remove]

def preset_to_config(name):
    verbose_print("+"*42)
    global model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df

    temp = get_preset_dict(name)
    verbose_print(temp)

    model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df = copy.deepcopy(temp[name])

    verbose_print(f"model_config_df:\t{model_config_df}")
    verbose_print(f"dataset_config_df:\t{dataset_config_df}")
    verbose_print(f"system_config_df:\t{system_config_df}")
    verbose_print(f"image_gen_config_df:\t{image_gen_config_df}")
    verbose_print(f"train_config_df:\t{train_config_df}")
    verbose_print("%"*42)

def change_regularizer_view(choice):
    if "Custom" in choice:
        image_gen_config_df["regularizer_var"] = 0
    elif "Auto" in choice:
        image_gen_config_df["regularizer_var"] = 1

def image_gen_config_save_button(final_img_path, seed_var, ddim_eta_var, scale_var, prompt_string, n_samples, n_iter, ddim_steps, keep_jpgs):
    global dataset_config_df, image_gen_config_df
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
    update_JSON(False)

    # create directories if necessary
    create_data_dirs()

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'subdirs:\t{sub_dir_names}')

    return final_img_path, gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])

def image_generation_button(keep_jpgs, presets_run_checkbox_group_var):
    verbose_print(f"===========------- IMAGE GENERATING -------===========")
    global model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df
    model_config_df_backup, dataset_config_df_backup, system_config_df_backup, image_gen_config_df_backup, train_config_df_backup = copy.deepcopy([model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df])

    if presets_run_checkbox_group_var and not presets_run_checkbox_group_var == "":
        for selected_preset in presets_run_checkbox_group_var:
            verbose_print(f"NOW RUNNING PRESET:\t{selected_preset}")
            # load preset into config only
            preset_to_config(selected_preset)
            # generate images
            image_generation_button(get_preset_dict(selected_preset)[3]["keep_jpgs"])#image_gen_config_df["keep_jpgs"]
    else:
        try:
            ckpt = model_config_df['model_name']
            if ' ' in ckpt:
                ckpt = f"\'{ckpt}\'"
            image_gen_cmd = f"python scripts/stable_txt2img.py --seed {image_gen_config_df['seed_var']} " \
                            f"--ddim_eta {image_gen_config_df['ddim_eta_var']} --n_samples {image_gen_config_df['n_samples']} " \
                            f"--n_iter {image_gen_config_df['n_iter']} --scale {image_gen_config_df['scale_var']} " \
                            f"--ddim_steps {image_gen_config_df['ddim_steps']} --ckpt {ckpt} " \
                            f"--prompt \'{image_gen_config_df['prompt_string']}\' --outdir {image_gen_config_df['final_img_path']}"

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
        except Exception:
            verbose_print("Not all image generation configurations are set")
    # configure session back to original
    model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df = [model_config_df_backup, dataset_config_df_backup, system_config_df_backup, image_gen_config_df_backup, train_config_df_backup]
    verbose_print(f"model_config_df:\t{model_config_df}")
    verbose_print(f"dataset_config_df:\t{dataset_config_df}")
    verbose_print(f"system_config_df:\t{system_config_df}")
    verbose_print(f"image_gen_config_df:\t{image_gen_config_df}")
    verbose_print(f"train_config_df:\t{train_config_df}")
    verbose_print("%"*42)

def train_save_button(max_training_steps, batch_size, cpu_workers, model_path):
    global image_gen_config_df, train_config_df
    train_config_df['max_training_steps'] = int(max_training_steps)
    train_config_df['batch_size'] = int(batch_size)
    train_config_df['cpu_workers'] = int(cpu_workers)
    train_config_df['model_path'] = model_path if (model_path and not model_path == '') else None if (not 'model_path' in image_gen_config_df) else (image_gen_config_df['model_path'])

    # update json file
    update_JSON(False)

    # create directories if necessary
    create_data_dirs()

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'subdirs:\t{sub_dir_names}')

    return gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])

def train_resume_checkbox(checkbox):
    if checkbox:
        if 'model_path' in train_config_df:
            return gr.update(label='Path to ckpt model in logs directory', value=str(train_config_df['model_path']), visible=True)
        else:
            return gr.update(label='Path to ckpt model in logs directory', visible=True)
    else:
        return gr.update(visible=False)


def get_full_model_paths(model_name_list):
    list_of_models = []

    model_options = []
    model_options_path = os.path.join(os.getcwd(), 'logs/')
    if os.path.exists(model_options_path):
        model_options = [os.path.join(model_options_path, directory) for directory in os.listdir(model_options_path)]
    if "model_path" in train_config_df and (train_config_df["model_path"]) and (
            not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
        model_options.append(train_config_df["model_path"])

    for name in model_name_list:
        # find all matches
        for full_path in model_options:
            if name in full_path and not os.path.isdir(full_path):
                if '.ckpt' in name:
                    list_of_models.append(full_path)
                    break
            elif name in full_path:
                # ckpt file ?
                for file in os.listdir(full_path):
                    if '.ckpt' in file:
                        list_of_models.append(full_path)
                        break
                    # contains checkpoint dir ?
                    elif 'checkpoints' in file and os.path.isdir(os.path.join(full_path, file)):
                        list_of_models.append(os.path.join(os.path.join(full_path, 'checkpoints/'), 'last.ckpt'))
                        break
    return list_of_models


def prune_ckpt(prune_model_checkbox_group_var):
    list_of_models = get_full_model_paths(prune_model_checkbox_group_var)
    for model_path in list_of_models:
        prune_cmd = f"python prune-ckpt.py --ckpt {model_path}"
        execute(prune_cmd)
        verbose_print(f"Model Pruning Complete!")

def train_button(train_resume_checkbox_var, presets_run_checkbox_group_var, resume_train_radio_var):
    verbose_print(f"===========------- TRAINING -------===========")
    global model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df
    model_config_df_backup, dataset_config_df_backup, system_config_df_backup, image_gen_config_df_backup, train_config_df_backup = copy.deepcopy([model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df])

    if presets_run_checkbox_group_var and not presets_run_checkbox_group_var == "":
        for selected_preset in presets_run_checkbox_group_var:
            verbose_print(f"NOW RUNNING PRESET:\t{selected_preset}")
            # load preset into config only
            preset_to_config(selected_preset)
            # train
            train_button(train_resume_checkbox_var=train_resume_checkbox_var, presets_run_checkbox_group_var="", resume_train_radio_var=resume_train_radio_var, )
    else:
        try:
            # train the model
            ckpt = model_config_df['model_name']
            if ' ' in ckpt:
                ckpt = f"\'{ckpt}\'"
            train_cmd = f"python main.py --base {dataset_config_df['config_path']} -t"

            if train_resume_checkbox_var and not resume_train_radio_var == "":
                list_of_models = get_full_model_paths([resume_train_radio_var])
                list_of_models = list_of_models[0] # there can only be one
                train_cmd = f"{train_cmd} --resume --actual_resume {list_of_models}"
            else:
                train_cmd = f"{train_cmd} --actual_resume {ckpt}"

            if image_gen_config_df['final_img_path'] and not image_gen_config_df['final_img_path'] == "":
                train_cmd = f"{train_cmd} --reg_data_root {image_gen_config_df['final_img_path']}"

            train_cmd = f"{train_cmd} -n {dataset_config_df['project_name']} " \
            f"--gpus {system_config_df['gpu_used_var']}, --data_root {dataset_config_df['dataset_path']} " \
            f"--max_training_steps {train_config_df['max_training_steps']} --class_word {image_gen_config_df['prompt_string']} --token {dataset_config_df['class_token']} " \
            f"--no-test --batch_size {train_config_df['batch_size']} --workers {train_config_df['cpu_workers']}"



            verbose_print("============================== TRAINING COMMAND TEST ==============================")
            verbose_print(train_cmd)
            verbose_print("============================== --------------------- ==============================")

            checklist_text = ["config_path", "model_name", "final_img_path", "project_name", "gpu_used_var", "dataset_path", "max_training_steps", "prompt_string"]
            run_checklist = [('config_path' in dataset_config_df), ('model_name' in model_config_df),
                             ('final_img_path' in image_gen_config_df), ('project_name' in dataset_config_df),
                             ('gpu_used_var' in system_config_df), ('dataset_path' in dataset_config_df),
                             ('max_training_steps' in train_config_df), ('prompt_string' in image_gen_config_df)]

            if all(run_checklist):
                for line in execute(train_cmd.split(" ")):
                    verbose_print(line)
            else:
                unmet_requirements = [i for i in range(len(run_checklist)) if not run_checklist[i]]
                unmet_requirements = [checklist_text[i] for i in range(len(unmet_requirements))]
                verbose_print(f"unmet_requirements:\t{unmet_requirements}")

        except Exception:
            verbose_print("Not all training configurations are set")
            verbose_print("If this is NOT a configuration error, then try entering the following command in the terminal:\nexport MKL_SERVICE_FORCE_INTEL=1")
    # configure session back to original
    model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df = [model_config_df_backup, dataset_config_df_backup, system_config_df_backup, image_gen_config_df_backup, train_config_df_backup]
    verbose_print(f"model_config_df:\t{model_config_df}")
    verbose_print(f"dataset_config_df:\t{dataset_config_df}")
    verbose_print(f"system_config_df:\t{system_config_df}")
    verbose_print(f"image_gen_config_df:\t{image_gen_config_df}")
    verbose_print(f"train_config_df:\t{train_config_df}")
    verbose_print("%"*42)

    model_options = []
    model_options_path = os.path.join(os.getcwd(), 'logs/')
    if os.path.exists(model_options_path):
        model_options = [os.path.join(model_options_path, directory) for directory in os.listdir(model_options_path)]
    if "model_path" in train_config_df and (train_config_df["model_path"]) and (
            not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
        model_options.append(train_config_df["model_path"])

    prune_model_checkbox_group_var = gr.update(label='Prune Model Options',
                                               choices=sorted([name.split("/")[-1] for name in model_options]),
                                               visible=bool(prune_model_checkbox_var))
    resume_train_radio_var = gr.update(label='Resume Model Train Options',
                                       choices=sorted([name.split("/")[-1] for name in model_options]),
                                       visible=False)
    return prune_model_checkbox_group_var, resume_train_radio_var

def merge_data_button(merge_data_list_var):
    global dataset_merge_dirs

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'sub_dir_names:\t{sub_dir_names}')

    paths_to_merge = []
    for name in merge_data_list_var:
        paths_to_merge.append(dataset_merge_dirs[sub_dir_names.index(name)])

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
    merge_data_list_var = gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])
    return merge_data_list_var

def create_new_dirs(paths, addon):
    new_paths = []
    for i in range(0, len(paths)):
        dataset = paths[i] + addon
        if not os.path.exists(dataset):
            new_paths.append(dataset)
            verbose_print(f"NEW dataset:\t{dataset}")
            dir_create_str = f"mkdir -p {dataset}"
            sub.run(dir_create_str.split(" "))
        else:
            verbose_print(f"dataset EXISTS:\t{dataset}")
    return new_paths

def horizontal_flip_button(merge_data_list_var):
    global dataset_merge_dirs
    # get all directories in the path
    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'sub_dir_names:\t{sub_dir_names}')
    # get selected paths
    paths_to_flip = []
    for name in merge_data_list_var:
        paths_to_flip.append(dataset_merge_dirs[sub_dir_names.index(name)])
    verbose_print(f'directories to generate flipped images:\t{paths_to_flip}')

    # generate new directories
    new_paths_to_flip = create_new_dirs(paths_to_flip, "_hflip")

    # loop for all sub-folders (SRC)
    for i in range(0, len(paths_to_flip)):
        ### get images from sub-dir (SRC)
        images_list = glob.glob1(paths_to_flip[i], "*.png")

        ### create & move images from each (SRC) into the (DST)
        counter = 0
        for image in images_list:
            original_image_path = os.path.join(paths_to_flip[i], image)
            # flip image
            img = Image.open(original_image_path)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            # save image
            img.save(os.path.join(new_paths_to_flip[i], f"{counter}.png"))
            counter += 1
        verbose_print(f'Done Flipping Images {paths_to_flip[i]} into {new_paths_to_flip[i]}')

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    merge_data_list_var = gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])
    return merge_data_list_var

def partial_image_crop_button(merge_data_list_var):
    global dataset_merge_dirs
    # get all directories in the path
    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    verbose_print(f'sub_dir_names:\t{sub_dir_names}')
    # get selected paths
    paths_to_flip = []
    for name in merge_data_list_var:
        paths_to_flip.append(dataset_merge_dirs[sub_dir_names.index(name)])
    verbose_print(f'directories to generate image fragments:\t{paths_to_flip}')

    # sample image & determine valid dimensions & ( portrait or landscape )
    images_list = glob.glob1(paths_to_flip[0], "*.png")
    original_image_path = os.path.join(paths_to_flip[0], images_list[0])
    img = Image.open(original_image_path)
    width, height = img.size
    if (width == 1024 and height == 512) or (width == 512 and height == 1024): # valid
        img_fragment_text = []
        crop_dims = [] # contains left, top, right, bottom (FOR EACH FRAGMENT)
        if (width == 512 and height == 1024): # portrait
            img_fragment_text = ["_top", "_mid", "_bottom"]
            mid_upper = math.ceil(height/3)
            mid_lower = mid_upper+512
            crop_dims = [(0, 0, width, int(height/2)), (0, mid_upper, width, mid_lower), (0, int(height/2), width, height)]
        else: # landscape
            img_fragment_text = ["_left", "_mid", "_right"]
            mid_left = math.ceil(width/3)
            mid_right = mid_left+512
            crop_dims = [(0, 0, int(width/2), height), (mid_left, 0, mid_right, height), (int(width/2), 0, width, height)]
        # generate new directories
        for fragment, crop_dim in zip(img_fragment_text, crop_dims):
            new_paths_to_flip = create_new_dirs(paths_to_flip, fragment)

            # loop for all sub-folders (SRC)
            for i in range(0, len(paths_to_flip)):
                ### get images from sub-dir (SRC)
                images_list = glob.glob1(paths_to_flip[i], "*.png")

                ### create & move images from each (SRC) into the (DST)
                counter = 0
                for image in images_list:
                    original_image_path = os.path.join(paths_to_flip[i], image)
                    # crop image
                    img = Image.open(original_image_path)
                    img = img.crop(crop_dim)
                    # save image
                    img.save(os.path.join(new_paths_to_flip[i], f"{counter}.png"))
                    counter += 1
                verbose_print(f'Done Generating Images of the \'{(fragment).split("_")}\' Fragment :: {paths_to_flip[i]} into {new_paths_to_flip[i]}')
    else:
        verbose_print(f"INVALID image resolutions detected. EXPECTED (512x1024) or (1024x512)")

    # update merged dirs list
    dataset_merge_dirs = update_merged_dirs()

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    merge_data_list_var = gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])
    return merge_data_list_var

def generate_random_long():
    import random
    return str(random.choice(range(0,10000000)))

def preset_save_button(preset_name_var, presets_load_dropdown_var, presets_delete_ckbx_var, presets_run_ckbx_var):
    global model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df
    global presets
    verbose_print(f"==========----- PRESET SAVE -----==========")
    # if preset has a name, use it to set the settings of that preset, then update json
    verbose_print("#"*42)
    verbose_print("#"*42)
    keys = get_all_presets_keys()
    verbose_print(f"preset_name_var:\t{preset_name_var}")
    verbose_print(f"keys:\t{keys}")
    verbose_print("#"*42)
    verbose_print("#"*42)

    if preset_name_var and not preset_name_var == "" and preset_name_var in keys:
        # update components: dropdown, all preset checkbox groups
        model_config_df["preset_name"] = preset_name_var

        temp_dictionary = {}

        # make copies of the gui config params as to not affect the presets later on
        model_config_df_temp, dataset_config_df_temp, system_config_df_temp, image_gen_config_df_temp, train_config_df_temp = copy.deepcopy([model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df])

        temp_dictionary[preset_name_var] = [model_config_df_temp, dataset_config_df_temp, system_config_df_temp, image_gen_config_df_temp, train_config_df_temp]
        index_to_remove = get_preset_index(preset_name_var)
        verbose_print(f"BEFORE:::\tlen(presets):\t{len(presets)}")
        verbose_print(f"temp_dictionary:\t{temp_dictionary}")
        verbose_print(f"index_to_remove:\t{index_to_remove}")
        remove_preset_from_list(preset_name_var)
        presets.append(temp_dictionary)
        verbose_print(f"presets[-1]:\t{presets[-1]}")
        verbose_print(f"presets:\t{presets}")
        verbose_print(f"AFTER:::\tlen(presets):\t{len(presets)}")

        # json update
        update_JSON(True)
        # update components: dropdown, all preset checkbox groups
        verbose_print(f"==========----- PRESET SAVE -----========== done")
        return gr.update(value=preset_name_var), gr.update(choices=list(get_all_presets_keys()), label='Optional Presets'), \
               gr.update(choices=list(get_all_presets_keys()), label='All Presets', visible=presets_delete_ckbx_var), \
               gr.update(choices=list(get_all_presets_keys()), label='All Presets', visible=presets_run_ckbx_var)
    else:
        verbose_print(f"NOT SAVED!!!  PLEASE FILL IN THE TEXTBOX FOR \'PRESET NAME\' and make sure the preset name matchs a preset name available")
        verbose_print(f"==========----- PRESET SAVE -----========== done")
        return preset_name_var, presets_load_dropdown_var, presets_delete_ckbx_var, presets_run_ckbx_var

def preset_add_button(preset_name_var, presets_delete_ckbx_var, presets_run_ckbx_var):
    global model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df
    global presets
    verbose_print(f"==========----- PRESET ADD -----==========")
    # if preset has a name, use it to set the settings of that preset, then update json
    # if no name, create a random long number for the name, and do the last two steps above
    temp_dictionary = {}
    temp_name = preset_name_var
    if not preset_name_var or preset_name_var == "":
        # create a random name
        temp_name = generate_random_long()

    # update components: dropdown, all preset checkbox groups
    model_config_df["preset_name"] = temp_name

    if not temp_name in get_all_presets_keys():
        verbose_print(f"BEFORE:::\tlen(presets):\t{len(presets)}")
        verbose_print(f"BEFORE:::\tpresets:\t{presets}")

        # make copies of the gui config params as to not affect the presets later on
        model_config_df_temp, dataset_config_df_temp, system_config_df_temp, image_gen_config_df_temp, train_config_df_temp = copy.deepcopy([model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df])

        # make a new preset
        temp_dictionary[temp_name] = [model_config_df_temp, dataset_config_df_temp, system_config_df_temp, image_gen_config_df_temp, train_config_df_temp]
        verbose_print(f"temp_dictionary:\t{temp_dictionary}")
        presets.append(temp_dictionary)
        verbose_print(f"presets[-1]:\t{presets[-1]}")
        verbose_print(f"AFTER:::presets:\t{presets}")
        verbose_print(f"AFTER:::\tlen(presets):\t{len(presets)}")
        # json update
        update_JSON(True)
    verbose_print(f"==========----- PRESET ADD -----========== done")

    return gr.update(value=preset_name_var), gr.update(choices=get_all_presets_keys(), label='Optional Presets'), \
           gr.update(choices=get_all_presets_keys(), label='All Presets', visible=presets_delete_ckbx_var), \
           gr.update(choices=get_all_presets_keys(), label='All Presets', visible=presets_run_ckbx_var)
def presets_delete_checkbox(presets_delete_ckbx_var):
    presets_delete_button_var = gr.update(value='Delete', variant='secondary', visible=presets_delete_ckbx_var)
    presets_delete_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', visible=presets_delete_ckbx_var)
    return presets_delete_button_var, presets_delete_checkbox_group_var

def presets_run_checkbox(presets_run_ckbx_var):
    presets_run_button_var = gr.update(value='Image Gen + Train', variant='primary', visible=presets_run_ckbx_var)
    presets_run_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', visible=presets_run_ckbx_var)
    return presets_run_button_var, presets_run_checkbox_group_var

def generate_and_train_button(presets_run_checkbox_group_var):
    verbose_print(f"==========----- GENERATE & TRAIN -----==========")
    verbose_print(f"BEFORE:::\tpresets_run_checkbox_group_var:\t{presets_run_checkbox_group_var}")
    all_preset_keys = get_all_presets_keys()
    verbose_print(f"all_preset_keys:\t{all_preset_keys}")
    remove_list = []
    for key_name in presets_run_checkbox_group_var:
        if not key_name in all_preset_keys:
            remove_list.append(key_name)
    for item in remove_list:
        presets_run_checkbox_group_var.remove(item)
    verbose_print(f"AFTER:::\tpresets_run_checkbox_group_var:\t{presets_run_checkbox_group_var}")

    # use the selected presets to in-order run everything
    image_generation_button(bool(image_gen_config_df["keep_jpgs"]), presets_run_checkbox_group_var)
    prune_btn = train_button(False, presets_run_checkbox_group_var)
    verbose_print(f"==========----- GENERATE & TRAIN -----========== done")
    return prune_btn

def presets_delete_button(presets_delete_checkbox_group_var, presets_delete_ckbx_var, presets_run_ckbx_var):
    verbose_print(f"==========----- PRESET DELETE -----==========")
    verbose_print(f"BEFORE:::\tpresets_delete_checkbox_group_var:\t{presets_delete_checkbox_group_var}")
    all_preset_keys = get_all_presets_keys()
    verbose_print(f"all_preset_keys:\t{all_preset_keys}")
    remove_list = []
    for key_name in presets_delete_checkbox_group_var:
        if not key_name in all_preset_keys:
            remove_list.append(key_name)
    for item in remove_list:
        presets_delete_checkbox_group_var.remove(item)
    verbose_print(f"AFTER:::\tpresets_delete_checkbox_group_var:\t{presets_delete_checkbox_group_var}")

    ###remove elements from these:
    for preset_key in presets_delete_checkbox_group_var:
        remove_preset_from_list(preset_key)
    # presets_delete_checkbox_group_var -choices
    presets_load_dropdown_var = gr.update(choices=get_all_presets_keys(), label='Optional Presets')
    presets_delete_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', visible=presets_delete_ckbx_var)
    presets_run_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', visible=presets_run_ckbx_var)
    #affected elements
    ###reset
    ###### remove preset from cache & json
    update_JSON(True)
    verbose_print(f"==========----- PRESET DELETE -----========== done")
    return presets_load_dropdown_var, presets_delete_checkbox_group_var, presets_run_checkbox_group_var

def presets_clear_button():
    verbose_print(f"==========----- PRESET CLEAR -----==========")
    # load all components with the DEFAULT
    presets_delete_ckbx_var = gr.update(label='Delete Presets', value=False)
    presets_delete_button_var = gr.update(value='Delete', variant='secondary', visible=False)
    presets_delete_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', value=[], visible=presets_delete_ckbx_var)
    model_var = gr.update(choices=ckpt_files, label='Select Model', value='')
    preset_name_var = gr.update(lines=1, label='Preset Name (Optional)', value='')
    project_name = gr.update(lines=1, label='Project Name', value='')
    class_token = gr.update(lines=1, label='Token (e.g. firstnamelastname)', value='')
    config_path = gr.update(lines=1, label='Path to Model YAML Config', value=os.path.join(os.getcwd(), 'configs/stable-diffusion/v1-finetune_unfrozen.yaml'))
    dataset_path = gr.update(lines=1, label='Path to Class Target Dataset', value='')
    reg_dataset_path = gr.update(lines=1, label='Path to Regularization Dataset', value='')
    model_config_df["verbose"] = False
    verbose = gr.update(label='Verbose Mode', value=False, interactive=True)

    temp_text = []
    gpu_list = [i for i in range(0, torch.cuda.device_count())]
    gpu_used_var = None

    if len(gpu_list) > 0:
        temp_text = [f"gpu: {i}" for i in range(0, torch.cuda.device_count())]
        if "gpu_used_var" in system_config_df:
            gpu_used_var = gr.Radio(choices=temp_text, value=temp_text[(system_config_df["gpu_used_var"])],
                                    label='Select GPU', visible=True)
        else:
            system_config_df["gpu_used_var"] = gpu_list[0]  # EXPECT THIS TO CHANGE IN THE FUTURE
            gpu_used_var = gr.Radio(choices=temp_text, label='Select GPU', visible=True)
    else:
        gpu_used_var = gr.Radio(choices=temp_text, label='Select GPU', visible=False)

    presets_run_ckbx_var = gr.update(label='Job Scheduler', value=False)
    presets_run_button_var = gr.update(value='Image Gen + Train', variant='primary', visible=False)
    presets_run_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', value=[], visible=presets_run_ckbx_var)
    reg_options = ['Custom (REG) Images (🔺 stylization, 🔻 robustness)',
                   'Auto (REG) Images (🔺 robustness,🔻 stylization)']
    regularizer_var = gr.update(choices=reg_options, label='Select Regularization Approach', value='')
    final_img_path = gr.update(lines=1, label='Regularization Data Path', value='')
    seed_var = gr.update(lines=1, label='Seed Number (int)', value=str(10))
    ddim_eta_var = gr.update(lines=1, label='DDIM eta (float)', value=str(0.0))
    scale_var = gr.update(lines=1, label='Scale (float)', value=str(10.0))
    prompt_string = gr.update(lines=1, label='Regularization Prompt (e.g. Person)', value='')
    image_gen_config_df["keep_jpgs"] = False
    keep_jpgs = gr.update(label='Keep JPGs', value=False)
    n_samples = gr.update(minimum=0, maximum=200, step=1, value=1, label='Samples per Iteration (i.e. batch size)')
    n_iter = gr.update(minimum=0, maximum=1000, step=10, value=200, label='Number of Iterations')
    ddim_steps = gr.update(minimum=0, maximum=250, step=5, value=50, label='Sampler Steps')
    max_training_steps = gr.update(minimum=0, maximum=20000, step=20, label='Max Training Steps', value=2000)
    batch_size = gr.update(minimum=1, maximum=64, step=1, label='Batch Size', value=1)
    cpu_workers = gr.update(minimum=1, maximum=mp.cpu_count(), step=1, label='Worker Threads',
                            value=int(mp.cpu_count() / 2))
    model_options = []
    model_options_path = os.path.join(os.getcwd(), 'logs/')
    if os.path.exists(model_options_path):
        model_options = [os.path.join(model_options_path, directory) for directory in os.listdir(model_options_path)]
    if "model_path" in train_config_df and (train_config_df["model_path"]) and (
            not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
        model_options.append(train_config_df["model_path"])

    prune_model_checkbox_var = gr.update(label='Prune Model', value=False)
    prune_model_checkbox_group_var = gr.update(label='Prune Model Options',
                                                      choices=sorted([name.split("/")[-1] for name in model_options]),
                                                      visible=False, value=[])
    train_resume_checkbox_var = gr.update(label='Resume Training (Uses the Current Model Path)', value=False)
    resume_train_radio_var = gr.update(label='Resume Model Train Options',
                                       choices=sorted([name.split("/")[-1] for name in model_options]),
                                       visible=False, value="")

    prune_model_var = gr.update(visible=False)
    model_path_var = gr.update(visible=False)

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    merge_data_list_var = gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])

    verbose_print(f"==========----- PRESET CLEAR -----========== done")

    return presets_delete_ckbx_var,presets_delete_button_var,presets_delete_checkbox_group_var,model_var,\
           preset_name_var,project_name,class_token,config_path,dataset_path,reg_dataset_path,\
           verbose,gpu_used_var,presets_run_ckbx_var,presets_run_button_var,presets_run_checkbox_group_var,\
           regularizer_var,final_img_path,seed_var,ddim_eta_var,scale_var,prompt_string,\
           keep_jpgs,n_samples,n_iter,ddim_steps,max_training_steps,batch_size,cpu_workers,prune_model_var,\
           train_resume_checkbox_var,model_path_var,merge_data_list_var, prune_model_checkbox_var, prune_model_checkbox_group_var, \
           resume_train_radio_var

def presets_load_button(presets_load_dropdown_var, presets_delete_ckbx_var, presets_run_ckbx_var, prune_model_checkbox_var, train_resume_checkbox_var):
    verbose_print(f"==========----- PRESET LOAD -----==========")
    presets_delete_ckbx_var_temp = presets_delete_ckbx_var
    presets_run_ckbx_var_temp = presets_run_ckbx_var

    global presets
    verbose_print(f"presets_load_dropdown_var:\t{presets_load_dropdown_var}")
    verbose_print(f"presets[presets_load_dropdown_var]:\t{presets[get_preset_index(presets_load_dropdown_var)]}")

    # load config
    preset_to_config(presets_load_dropdown_var)

    verbose_print(f"547895897352978023908723978032987032978035978034978049872039870239870359870349870239870398702398703498702398702398705234983254")
    verbose_print(f"model_config_df:\t{model_config_df}")
    verbose_print(f"dataset_config_df:\t{dataset_config_df}")
    verbose_print(f"system_config_df:\t{system_config_df}")
    verbose_print(f"image_gen_config_df:\t{image_gen_config_df}")
    verbose_print(f"train_config_df:\t{train_config_df}")
    verbose_print(f"547895897352978023908723978032987032978035978034978049872039870239870359870349870239870398702398703498702398702398705234983254")


    presets_load_dropdown_var,presets_delete_ckbx_var, presets_delete_button_var, presets_delete_checkbox_group_var, model_var, \
    preset_name_var, project_name, class_token, config_path, dataset_path, reg_dataset_path, \
    verbose, gpu_used_var, presets_run_ckbx_var, presets_run_button_var, presets_run_checkbox_group_var, \
    regularizer_var, final_img_path, seed_var, ddim_eta_var, scale_var, prompt_string, \
    keep_jpgs, n_samples, n_iter, ddim_steps, max_training_steps, batch_size, cpu_workers, prune_model_var, \
    train_resume_checkbox_var, model_path_var, merge_data_list_var = [None]*33

    # update components
    presets_load_dropdown_var = gr.update(choices=get_all_presets_keys(), label='Optional Presets')
    presets_delete_ckbx_var = gr.update(label='Delete Presets', value=presets_delete_ckbx_var_temp)
    presets_delete_button_var = gr.update(value='Delete', variant='secondary', visible=False)
    presets_delete_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', value=[], visible=presets_delete_ckbx_var_temp)

    preset_name_var = gr.update(lines=1, value=model_config_df["preset_name"], label='Preset Name (Optional)')

    if "model_name" in model_config_df:
        model_var = gr.update(choices=ckpt_files, value=str(model_config_df["model_name"]),
                                    label='Select Model')
    else:
        model_var = gr.update(choices=ckpt_files, label='Select Model')
    if "project_name" in dataset_config_df:
        project_name = gr.update(lines=1, label='Project Name',
                                  value=str(dataset_config_df["project_name"]))
    else:
        project_name = gr.update(lines=1, label='Project Name')
    if "class_token" in dataset_config_df:
        class_token = gr.update(lines=1, label='Token (e.g. firstnamelastname)',
                                 value=str(dataset_config_df["class_token"]))
    else:
        class_token = gr.update(lines=1, label='Token (e.g. firstnamelastname)')
    if "config_path" in dataset_config_df:
        config_path = gr.update(lines=1, label='Path to Model YAML Config',
                                 value=str(dataset_config_df["config_path"]))
    else:
        config_path = gr.update(lines=1, label='Path to Model YAML Config', value=os.path.join(os.getcwd(), 'configs/stable-diffusion/v1-finetune_unfrozen.yaml'))
    if "dataset_path" in dataset_config_df:
        dataset_path = gr.update(lines=1, label='Path to Class Target Dataset',
                                  value=str(dataset_config_df["dataset_path"]))
    else:
        dataset_path = gr.update(lines=1, label='Path to Class Target Dataset')
    if "reg_dataset_path" in dataset_config_df:
        reg_dataset_path = gr.update(lines=1, label='Path to Regularization Dataset',
                                      value=str(dataset_config_df["reg_dataset_path"]))
    else:
        reg_dataset_path = gr.update(lines=1, label='Path to Regularization Dataset')
    if "verbose" in model_config_df:
        verbose = gr.update(label='Verbose Mode', value=bool(model_config_df["verbose"]), interactive=True)
    else:
        model_config_df["verbose"] = False
        verbose = gr.update(label='Verbose Mode', value=False, interactive=True)

    temp_text = []
    gpu_list = [i for i in range(0, torch.cuda.device_count())]
    gpu_used_var = None

    if len(gpu_list) > 0:
        temp_text = [f"gpu: {i}" for i in range(0, torch.cuda.device_count())]
        if "gpu_used_var" in system_config_df:
            gpu_used_var = gr.update(choices=temp_text, value=temp_text[(system_config_df["gpu_used_var"])],
                                    label='Select GPU', visible=True)
        else:
            system_config_df["gpu_used_var"] = gpu_list[0]  # EXPECT THIS TO CHANGE IN THE FUTURE
            gpu_used_var = gr.update(choices=temp_text, label='Select GPU', visible=True)
    else:
        gpu_used_var = gr.update(choices=temp_text, label='Select GPU', visible=False)

    presets_run_ckbx_var = gr.update(label='Job Scheduler', value=presets_run_ckbx_var_temp)
    presets_run_button_var = gr.update(value='Image Gen + Train', variant='primary', visible=False)
    presets_run_checkbox_group_var = gr.update(choices=get_all_presets_keys(), label='All Presets', value=[], visible=presets_run_ckbx_var_temp)
    reg_options = ['Custom (REG) Images (🔺 stylization, 🔻 robustness)',
                   'Auto (REG) Images (🔺 robustness,🔻 stylization)']
    if "regularizer_var" in image_gen_config_df:
        regularizer_var = gr.update(choices=reg_options,
                                          value=reg_options[(image_gen_config_df["regularizer_var"])],
                                          label='Select Regularization Approach')
    else:
        regularizer_var = gr.update(choices=reg_options,
                                          label='Select Regularization Approach')
    if not "final_img_path" in image_gen_config_df and not "reg_dataset_path" in dataset_config_df:
        final_img_path = gr.update(lines=1, label='Regularization Data Path')
    elif "reg_dataset_path" in dataset_config_df:
        final_img_path = gr.update(lines=1, label='Regularization Data Path',
                                    value=str(dataset_config_df["reg_dataset_path"]))
    elif "final_img_path" in image_gen_config_df:
        final_img_path = gr.update(lines=1, label='Regularization Data Path',
                                    value=str(image_gen_config_df["final_img_path"]))
    if "seed_var" in image_gen_config_df:
        seed_var = gr.update(lines=1, label='Seed Number (int)',
                              value=str(image_gen_config_df["seed_var"]))
    else:
        seed_var = gr.update(lines=1, label='Seed Number (int)', value=str(10))
    if "ddim_eta_var" in image_gen_config_df:
        ddim_eta_var = gr.update(lines=1, label='DDIM eta (float)',
                                  value=str(image_gen_config_df["ddim_eta_var"]))
    else:
        ddim_eta_var = gr.update(lines=1, label='DDIM eta (float)', value=str(0.0))
    if "scale_var" in image_gen_config_df:
        scale_var = gr.update(lines=1, label='Scale (float)',
                               value=str(image_gen_config_df["scale_var"]))
    else:
        scale_var = gr.update(lines=1, label='Scale (float)', value=str(10.0))
    if "prompt_string" in image_gen_config_df:
        prompt_string = gr.update(lines=1, label='Regularization Prompt (e.g. Person)',
                                   value=str(image_gen_config_df["prompt_string"]))
    else:
        prompt_string = gr.update(lines=1, label='Regularization Prompt (e.g. Person)')
    if "keep_jpgs" in image_gen_config_df:
        keep_jpgs = gr.update(label='Keep JPGs', value=bool(image_gen_config_df["keep_jpgs"]))
    else:
        image_gen_config_df["keep_jpgs"] = False
        keep_jpgs = gr.update(label='Keep JPGs', value=False)
    if "n_samples" in image_gen_config_df:
        n_samples = gr.update(minimum=0, maximum=200, step=1, label='Samples per Iteration (i.e. batch size)',
                              value=int(image_gen_config_df["n_samples"]))
    else:
        n_samples = gr.update(minimum=0, maximum=200, step=1, value=1, label='Samples per Iteration (i.e. batch size)')
    if "n_iter" in image_gen_config_df:
        n_iter = gr.update(minimum=0, maximum=1000, step=10, label='Number of Iterations',
                           value=int(image_gen_config_df["n_iter"]))
    else:
        n_iter = gr.update(minimum=0, maximum=1000, step=10, value=200, label='Number of Iterations')
    if "ddim_steps" in image_gen_config_df:
        ddim_steps = gr.update(minimum=0, maximum=250, step=5, label='Sampler Steps',
                               value=int(image_gen_config_df["ddim_steps"]))
    else:
        ddim_steps = gr.update(minimum=0, maximum=250, step=5, value=50, label='Sampler Steps')
    if "max_training_steps" in train_config_df:
        max_training_steps = gr.update(minimum=0, maximum=20000, step=20, label='Max Training Steps',
                                       value=int(train_config_df["max_training_steps"]))
    else:
        max_training_steps = gr.update(minimum=0, maximum=20000, step=20, label='Max Training Steps', value=2000)
    if "batch_size" in train_config_df:
        batch_size = gr.update(minimum=1, maximum=64, step=1, label='Batch Size',
                               value=int(train_config_df["batch_size"]))
    else:
        batch_size = gr.update(minimum=1, maximum=64, step=1, label='Batch Size', value=1)
    if "cpu_workers" in train_config_df:
        cpu_workers = gr.update(minimum=1, maximum=mp.cpu_count(), step=1, label='Worker Threads',
                                value=int(train_config_df["cpu_workers"]))
    else:
        cpu_workers = gr.update(minimum=1, maximum=mp.cpu_count(), step=1, label='Worker Threads',
                                value=int(mp.cpu_count() / 2))

    model_options = []
    model_options_path = os.path.join(os.getcwd(), 'logs/')
    if os.path.exists(model_options_path):
        model_options = [os.path.join(model_options_path, directory) for directory in os.listdir(model_options_path)]
    if "model_path" in train_config_df and (train_config_df["model_path"]) and (
    not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
        model_options.append(train_config_df["model_path"])

    prune_model_checkbox_var = gr.update(label='Prune Model', value=bool(prune_model_checkbox_var))
    prune_model_checkbox_group_var = gr.update(label='Prune Model Options', choices=sorted([name.split("/")[-1] for name in model_options]), visible=bool(prune_model_checkbox_var), value=[])
    train_resume_checkbox_var = gr.update(label='Resume Training (Uses the Current Model Path)', value=bool(train_resume_checkbox_var))
    resume_train_radio_var = gr.update(label='Resume Model Train Options',
                                       choices=sorted([name.split("/")[-1] for name in model_options]),
                                       visible=bool(train_resume_checkbox_var), value="")

    prune_model_var = gr.update(visible=bool(prune_model_checkbox_var))
    model_path_var = gr.update(visible=bool(prune_model_checkbox_var) or bool(train_resume_checkbox_var))

    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
    merge_data_list_var = gr.update(choices=sub_dir_names, label="Dataset Sub-Directories", value=[])

    verbose_print(f"==========----- PRESET LOAD -----========== done")

    return presets_load_dropdown_var, presets_delete_ckbx_var, presets_delete_button_var, presets_delete_checkbox_group_var, model_var, \
        preset_name_var, project_name, class_token, config_path, dataset_path, reg_dataset_path, \
        verbose, gpu_used_var, presets_run_ckbx_var, presets_run_button_var, presets_run_checkbox_group_var, \
        regularizer_var, final_img_path, seed_var, ddim_eta_var, scale_var, prompt_string, \
        keep_jpgs, n_samples, n_iter, ddim_steps, max_training_steps, batch_size, cpu_workers, prune_model_var, \
        train_resume_checkbox_var, model_path_var, merge_data_list_var, prune_model_checkbox_var, prune_model_checkbox_group_var, \
        resume_train_radio_var

def resume_train_radio(train_resume_checkbox_var):
    model_options = []
    model_options_path = os.path.join(os.getcwd(), 'logs/')
    if os.path.exists(model_options_path):
        model_options = [os.path.join(model_options_path, directory) for directory in os.listdir(model_options_path)]
    if "model_path" in train_config_df and (train_config_df["model_path"]) and (
    not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
        model_options.append(train_config_df["model_path"])
    resume_train_radio_var = gr.update(label='Resume Model Train Options', choices=sorted([name.split("/")[-1] for name in model_options]), visible=bool(train_resume_checkbox_var))
    return resume_train_radio_var

def prune_model_checkbox(prune_model_checkbox_var):
    model_options = []
    model_options_path = os.path.join(os.getcwd(), 'logs/')
    if os.path.exists(model_options_path):
        model_options = [os.path.join(model_options_path, directory) for directory in os.listdir(model_options_path)]
    if "model_path" in train_config_df and (train_config_df["model_path"]) and (
    not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
        model_options.append(train_config_df["model_path"])
    prune_model_checkbox_group_var = gr.update(label='Prune Model Options', choices=sorted([name.split("/")[-1] for name in model_options]), visible=bool(prune_model_checkbox_var))
    prune_model_var = gr.update(visible=bool(prune_model_checkbox_var))
    return prune_model_checkbox_group_var, prune_model_var

'''
##################################################################################################################################
################################################     GUI BLOCKS DEFINED BELOW     ################################################
##################################################################################################################################
'''
with gr.Blocks() as demo:
    with gr.Tab("Model & Data Configuration"):
        with gr.Row():
            config_save_var = gr.Button(value="Apply & Save Settings", variant='primary')
            preset_save_var = gr.Button(value="Update/Save (current loaded) Preset", variant='primary')
            preset_add_var = gr.Button(value="Add (current session) to Preset", variant='secondary')
            preset_clear_var = gr.Button(value="Clear (current loaded) Preset/Session", variant='secondary')
        gr.Markdown(
        """
        ### Make sure a stable diffusion model with the (.ckpt) extension has been downloaded
        ### Please move the downloaded model into the "Dreambooth-WebUI" repository folder
        #### Important Note: Please Save the \'Session\' Settings, before managing the \'Preset\' Button Options
        #### Please NO spaces in the ckpt file name
        """)

        verbose_print(f"ckpt_files {ckpt_files}")
        with gr.Row():
            dropdown_keys = get_all_presets_keys()
            verbose_print(f"dropdown_keys:\t{dropdown_keys}")
            presets_load_dropdown_var = gr.Dropdown(choices=dropdown_keys, label='Optional Presets')

        with gr.Row():
            with gr.Column():
                presets_delete_ckbx_var = gr.Checkbox(label='Delete Presets', value=False)
                presets_delete_button_var = gr.Button(value='Delete', variant='secondary', visible=False)
            with gr.Column():
                presets_delete_checkbox_group_var = gr.CheckboxGroup(choices=get_all_presets_keys(), label='All Presets', visible=False)

        with gr.Row():
            if "model_name" in model_config_df:
                model_var = gr.Radio(choices=ckpt_files, value=str(model_config_df["model_name"]), label='Select Model')
            else:
                model_var = gr.Radio(choices=ckpt_files, label='Select Model')
            if "preset_name" in model_config_df:
                preset_name_var = gr.Textbox(lines=1, value=str(model_config_df["preset_name"]), label='Preset Name (Optional)')
            else:
                preset_name_var = gr.Textbox(lines=1, label='Preset Name (Optional)')

        if "project_name" in dataset_config_df:
            project_name = gr.Textbox(lines=1, label='Project Name', value=str(dataset_config_df["project_name"]))
        else:
            project_name = gr.Textbox(lines=1, label='Project Name')
        if "class_token" in dataset_config_df:
            class_token = gr.Textbox(lines=1, label='Token (e.g. firstnamelastname)', value=str(dataset_config_df["class_token"]))
        else:
            class_token = gr.Textbox(lines=1, label='Token (e.g. firstnamelastname)')
        if "config_path" in dataset_config_df:
            config_path = gr.Textbox(lines=1, label='Path to Model YAML Config', value=str(dataset_config_df["config_path"]))
        else:
            config_path = gr.Textbox(lines=1, label='Path to Model YAML Config', value=os.path.join(os.getcwd(), 'configs/stable-diffusion/v1-finetune_unfrozen.yaml'))
        if "dataset_path" in dataset_config_df:
            dataset_path = gr.Textbox(lines=1, label='Path to Class Target Dataset', value=str(dataset_config_df["dataset_path"]))
        else:
            dataset_path = gr.Textbox(lines=1, label='Path to Class Target Dataset')
        if "reg_dataset_path" in dataset_config_df:
            reg_dataset_path = gr.Textbox(lines=1, label='Path to Regularization Dataset', value=str(dataset_config_df["reg_dataset_path"]))
        else:
            reg_dataset_path = gr.Textbox(lines=1, label='Path to Regularization Dataset')

        with gr.Row():
            with gr.Column():
                if "verbose" in model_config_df:
                    verbose = gr.Checkbox(label='Verbose Mode', value=bool(model_config_df["verbose"]), interactive=True)
                else:
                    model_config_df["verbose"] = False
                    verbose = gr.Checkbox(label='Verbose Mode', value=False, interactive=True)
            with gr.Column():
                temp_text = []
                gpu_list = [i for i in range(0, torch.cuda.device_count())]
                gpu_used_var = None

                if len(gpu_list) > 0:
                    temp_text = [f"gpu: {i}" for i in range(0, torch.cuda.device_count())]
                    if "gpu_used_var" in system_config_df:
                        gpu_used_var = gr.Radio(choices=temp_text, value=temp_text[(system_config_df["gpu_used_var"])],
                                                label='Select GPU', visible=True)
                    else:
                        system_config_df["gpu_used_var"] = gpu_list[0]  # EXPECT THIS TO CHANGE IN THE FUTURE
                        gpu_used_var = gr.Radio(choices=temp_text, label='Select GPU', visible=True)
                else:
                    gpu_used_var = gr.Radio(choices=temp_text, label='Select GPU', visible=False)


        with gr.Accordion("Trying to Schedule Multiple Experiments without intervention? (Look Here!)"):
            gr.Markdown(
                """
                ### Runs all \"CHECKED\" presets in an arbitrary order
                ### Allows for either \"Batched Image Generation Runs\"  OR  \"Batched Training Runs\" on their respective tab
                ### New Button below for \"Batched ( Image Gen + Training ) Runs\"
                """)
            with gr.Row():
                with gr.Column():
                    presets_run_ckbx_var = gr.Checkbox(label='Job Scheduler', value=False)
                    presets_run_button_var = gr.Button(value='Image Gen + Train', variant='primary', visible=False)
                with gr.Column():
                    presets_run_checkbox_group_var = gr.CheckboxGroup(choices=get_all_presets_keys(), label='All Presets', visible=False)

    with gr.Tab("Image Regularization"):
        gr.Markdown(
        """
        ### Please make sure that if you are using a custom regularization dataset, that the images are formatted 512x512
        ### Here is an online formatter for cropping images if needed [https://www.birme.net/](https://www.birme.net/)
        #### Important Note: Please Save the \'Session\' Settings, before managing the \'Preset\' Button Options
        #### Please NO spaces in the ckpt file name
        """)
        regularizer_save_var = gr.Button(value="Apply & Save Settings", variant='primary')
        with gr.Row():
            reg_options = ['Custom (REG) Images (🔺 stylization, 🔻 robustness)', 'Auto (REG) Images (🔺 robustness,🔻 stylization)']
            if "regularizer_var" in image_gen_config_df:
                regularizer_var = gr.Radio(choices=reg_options,
                                value=reg_options[(image_gen_config_df["regularizer_var"])], label='Select Regularization Approach')
            else:
                regularizer_var = gr.Radio(choices=reg_options,
                                label='Select Regularization Approach')
            if not "final_img_path" in image_gen_config_df and not "reg_dataset_path" in dataset_config_df:
                final_img_path = gr.Textbox(lines=1, label='Regularization Data Path')
            elif "reg_dataset_path" in dataset_config_df:
                final_img_path = gr.Textbox(lines=1, label='Regularization Data Path',
                                            value=str(dataset_config_df["reg_dataset_path"]))
            elif "final_img_path" in image_gen_config_df:
                final_img_path = gr.Textbox(lines=1, label='Regularization Data Path',
                                            value=str(image_gen_config_df["final_img_path"]))

        if "seed_var" in image_gen_config_df:
            seed_var = gr.Textbox(lines=1, label='Seed Number (int)', value=str(image_gen_config_df["seed_var"]))
        else:
            seed_var = gr.Textbox(lines=1, label='Seed Number (int)', value=str(10))
        if "ddim_eta_var" in image_gen_config_df:
            ddim_eta_var = gr.Textbox(lines=1, label='DDIM eta (float)', value=str(image_gen_config_df["ddim_eta_var"]))
        else:
            ddim_eta_var = gr.Textbox(lines=1, label='DDIM eta (float)', value=str(0.0))
        if "scale_var" in image_gen_config_df:
            scale_var = gr.Textbox(lines=1, label='Scale (float)', value=str(image_gen_config_df["scale_var"]))
        else:
            scale_var = gr.Textbox(lines=1, label='Scale (float)', value=str(10.0))

        with gr.Row():
            with gr.Column():
                if "prompt_string" in image_gen_config_df:
                    prompt_string = gr.Textbox(lines=1, label='Regularization Prompt (e.g. Person)', value=str(image_gen_config_df["prompt_string"]))
                else:
                    prompt_string = gr.Textbox(lines=1, label='Regularization Prompt (e.g. Person)')
            with gr.Column():
                if "keep_jpgs" in image_gen_config_df:
                    keep_jpgs = gr.Checkbox(label='Keep JPGs', value=bool(image_gen_config_df["keep_jpgs"]))
                else:
                    image_gen_config_df["keep_jpgs"] = False
                    keep_jpgs = gr.Checkbox(label='Keep JPGs', value=False)

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
        gr.Markdown(
        """
        #### Important Note: Please Save the \'Session\' Settings, before managing the \'Preset\' Button Options
        #### Please NO spaces in the ckpt file name
        """)
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

        model_options = []
        model_options_path = os.path.join(os.getcwd(), 'logs/')
        if os.path.exists(model_options_path):
            model_options = [os.path.join(model_options_path, directory) for directory in os.listdir(model_options_path)]
        if "model_path" in train_config_df and (train_config_df["model_path"]) and (not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
            model_options.append(train_config_df["model_path"])
        prune_model_checkbox_var = gr.Checkbox(label='Prune Model')
        prune_model_checkbox_group_var = gr.CheckboxGroup(label='Prune Model Options', choices=sorted([name.split("/")[-1] for name in model_options]), visible=False)

        with gr.Row():
            train_out_var = gr.Button(value="Train", variant='secondary')
            prune_model_var = gr.Button(value="Prune Model", visible=False)

        with gr.Accordion("Trying to Resume Training?  OR  Merge Data Directories  OR  Use Augmentation Methods? (Look Here!)"):
            gr.Markdown(
                """
                ### Make sure a you have the full checkpoint directory in the ( logs ) directory :: if resuming training
                ### Make sure to check all sub-directories you want merged within the path of your data directory
                ### ( IMPORTANT ) if merging, be aware that the data merged is (not copied); please back up sub-directories if you want to keep their contents!
                ### ( EVEN MORE IMPORTANT ) make sure that if merging sub-directories, that there is no left over data still in the current dataset_path. In other words make sure images are in some kind of sub-directory. (otherwise data could be overwritten!)
                ### To refresh the list of sub-directories (checkboxes) after merging, then go to the first tab in the UI and click ( APPLY SETTINGS )
                #### The Horizontal Flip \'Button\' creates new directories containing the flipped images
                #### The Partial Image Crop \'Button\' generates THREE new directories depending on if the image is landscape (1024x512) or portrait (512x1024) oriented
                #### The Partial Image Crop \'Button\' creates a sliding window crop over the image equidistant from each other
                #### It is IMPORTANT that images are either (1024x512) or (512x1024) to use the partial crop button
                """)
            with gr.Row():
                model_options = []
                model_options_path = os.path.join(os.getcwd(), 'logs/')
                if os.path.exists(model_options_path):
                    model_options = [os.path.join(model_options_path, directory) for directory in
                                     os.listdir(model_options_path)]
                if "model_path" in train_config_df and (train_config_df["model_path"]) and (not train_config_df["model_path"] == "") and (not train_config_df["model_path"] == "None"):
                    model_options.append(train_config_df["model_path"])
                train_resume_checkbox_var = gr.Checkbox(label='Resume Training (Uses the Current Model Path)')
                model_path_var = gr.Textbox(visible=False)
                resume_train_radio_var = gr.Radio(label='Resume Model Train Options', choices=sorted(
                    [name.split("/")[-1] for name in model_options]), visible=False)
            with gr.Row():
                with gr.Column():
                    sub_dir_names = [data_dir_path.split('/')[-1] for data_dir_path in dataset_merge_dirs]
                    merge_data_list_var = gr.CheckboxGroup(choices=sub_dir_names, label="Dataset Sub-Directories")
                with gr.Column():
                    merge_data_button_var = gr.Button(value="Merge All Data Sub-Directories", variant='secondary')
                    horizontal_flip_button_var = gr.Button(value="Horizontal Flip of Data in Sub-Directories", variant='secondary')
                    partial_image_crop_button_var = gr.Button(value="Partial Image Crop (makes 3 fragments of the original)", variant='secondary')
    with gr.Tab("Real-Time Loss Graphs"):
        gr.Markdown(
        """
        # Coming Soon!
        """)

    model_var.change(fn=model_choice, inputs=[model_var], outputs=[])
    config_save_var.click(fn=model_config_save_button,
                          inputs=[model_var,
                                  gpu_used_var,
                                  project_name,
                                  class_token,
                                  config_path,
                                  dataset_path,
                                  reg_dataset_path,
                                  preset_name_var
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

    generate_images_var.click(fn=image_generation_button, inputs=[keep_jpgs, presets_run_checkbox_group_var], outputs=[], show_progress=True, scroll_to_output=True)

    fine_tine_save_var.click(fn=train_save_button, inputs=[max_training_steps,
                                                                        batch_size,
                                                                        cpu_workers,
                                                                        model_path_var
                                                                        ], outputs=[merge_data_list_var])
    train_out_var.click(fn=train_button, inputs=[train_resume_checkbox_var, presets_run_checkbox_group_var, resume_train_radio_var], outputs=[prune_model_checkbox_group_var, resume_train_radio_var], show_progress=True, scroll_to_output=True)###### and output for  training UPDATE CHECKBOXGROUP
    prune_model_var.click(fn=prune_ckpt, inputs=[prune_model_checkbox_group_var], outputs=[])
    train_resume_checkbox_var.change(fn=train_resume_checkbox, inputs=[train_resume_checkbox_var], outputs=[model_path_var])

    merge_data_button_var.click(fn=merge_data_button, inputs=[merge_data_list_var], outputs=[merge_data_list_var])

    preset_save_var.click(fn=preset_save_button, inputs=[preset_name_var, presets_load_dropdown_var, presets_delete_ckbx_var, presets_run_ckbx_var],
                          outputs=[preset_name_var, presets_load_dropdown_var, presets_delete_checkbox_group_var, presets_run_checkbox_group_var])

    preset_add_var.click(fn=preset_add_button, inputs=[preset_name_var, presets_delete_ckbx_var, presets_run_ckbx_var],
                         outputs=[presets_load_dropdown_var, presets_delete_checkbox_group_var, presets_run_checkbox_group_var])

    presets_delete_ckbx_var.change(fn=presets_delete_checkbox, inputs=[presets_delete_ckbx_var], outputs=[presets_delete_button_var, presets_delete_checkbox_group_var])
    presets_run_ckbx_var.change(fn=presets_run_checkbox, inputs=[presets_run_ckbx_var], outputs=[presets_run_button_var, presets_run_checkbox_group_var])

    presets_run_button_var.click(fn=generate_and_train_button, inputs=[presets_run_checkbox_group_var], outputs=[prune_model_var])

    presets_delete_button_var.click(fn=presets_delete_button, inputs=[presets_delete_checkbox_group_var, presets_delete_ckbx_var, presets_run_ckbx_var],
                                    outputs=[presets_load_dropdown_var, presets_delete_checkbox_group_var, presets_run_checkbox_group_var])

    preset_clear_var.click(fn=presets_clear_button, inputs=[], outputs=[presets_delete_ckbx_var,presets_delete_button_var,presets_delete_checkbox_group_var,model_var,\
               preset_name_var,project_name,class_token,config_path,dataset_path,reg_dataset_path,\
               verbose,gpu_used_var,presets_run_ckbx_var,presets_run_button_var,presets_run_checkbox_group_var,\
               regularizer_var,final_img_path,seed_var,ddim_eta_var,scale_var,prompt_string,\
               keep_jpgs,n_samples,n_iter,ddim_steps,max_training_steps,batch_size,cpu_workers,prune_model_var,\
               train_resume_checkbox_var,model_path_var,merge_data_list_var, prune_model_checkbox_var, prune_model_checkbox_group_var, resume_train_radio_var])

    presets_load_dropdown_var.change(fn=presets_load_button, inputs=[presets_load_dropdown_var, presets_delete_ckbx_var, presets_run_ckbx_var, prune_model_checkbox_var, train_resume_checkbox_var], outputs=[presets_load_dropdown_var, presets_delete_ckbx_var, presets_delete_button_var, presets_delete_checkbox_group_var, model_var, \
                preset_name_var, project_name, class_token, config_path, dataset_path, reg_dataset_path, \
                verbose, gpu_used_var, presets_run_ckbx_var, presets_run_button_var, presets_run_checkbox_group_var, \
                regularizer_var, final_img_path, seed_var, ddim_eta_var, scale_var, prompt_string, \
                keep_jpgs, n_samples, n_iter, ddim_steps, max_training_steps, batch_size, cpu_workers, prune_model_var, \
                train_resume_checkbox_var, model_path_var, merge_data_list_var, prune_model_checkbox_var, prune_model_checkbox_group_var, resume_train_radio_var])

    horizontal_flip_button_var.click(fn=horizontal_flip_button, inputs=[merge_data_list_var], outputs=[merge_data_list_var])
    partial_image_crop_button_var.click(fn=partial_image_crop_button, inputs=[merge_data_list_var], outputs=[merge_data_list_var])

    prune_model_checkbox_var.change(fn=prune_model_checkbox, inputs=[prune_model_checkbox_var], outputs=[prune_model_checkbox_group_var, prune_model_var])
    train_resume_checkbox_var.change(fn=resume_train_radio, inputs=[train_resume_checkbox_var],
                                    outputs=[resume_train_radio_var])

if __name__ == "__main__":
    demo.launch()
