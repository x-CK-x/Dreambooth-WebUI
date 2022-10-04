import gradio as gr
import os
import sys
import subprocess as sub
import glob
import json

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

def dependency_install_button():
    return sub.run(f"pip install -r {cwd}/requirements.txt".split(" "), stdout=sub.PIPE).stdout.decode("utf-8")

def update_JSON():
    temp = [model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df]
    for entry in temp:
        print(entry)

    with open(json_file_name, "w") as f:
        json.dump([model_config_df, dataset_config_df, system_config_df, image_gen_config_df, train_config_df], indent=4, fp=f)
    f.close()

def create_data_dirs():
    if not os.path.exists(dataset_config_df["dataset_path"]):
        dataset = dataset_config_df["dataset_path"]
        dir_create_str = f"mkdir -p {dataset}"
        sub.run(dir_create_str.split(" "))
    if not os.path.exists(dataset_config_df["reg_dataset_path"]):
        dataset = dataset_config_df["reg_dataset_path"]
        dir_create_str = f"mkdir -p {dataset}"
        sub.run(dir_create_str.split(" "))

def model_choice(ver):
    for i in range(0, len(ckpt_files)):
        if ver == ckpt_files[i]:
            return ver
    return None

def model_config_save_button(model_name, gpu_used_var, project_name, class_word, config_path, dataset_path, reg_dataset_path):
    model_config_df["model_name"] = model_name
    system_config_df["gpu_used_var"] = int(gpu_used_var.replace("gpu: ", ""))
    dataset_config_df["project_name"] = project_name
    dataset_config_df["class_word"] = class_word
    dataset_config_df["config_path"] = config_path
    dataset_config_df["dataset_path"] = dataset_path
    dataset_config_df["reg_dataset_path"] = reg_dataset_path

    all_lines = None
    is_target_same = False
    with open(os.path.join(cwd, "ldm/data/personalized.py"), "r") as script_file:
        lines = script_file.readlines()
        for i in range(0, len(lines)):
            if "{}" in lines[i]:
                prior_class = (lines[i].split("\'")[1]).split(" ")[0]
                if prior_class == dataset_config_df["project_name"]:
                    is_target_same = True
                    break
                lines[i] = lines[i].replace(prior_class, dataset_config_df["project_name"])
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

def change_regularizer_view(choice):
    if "Custom" in choice:
        image_gen_config_df["regularizer_var"] = 0
    elif "Auto" in choice:
        image_gen_config_df["regularizer_var"] = 1

def image_gen_config_save_button(final_img_path, seed_var, ddim_eta_var, scale_var, prompt_name, n_samples, n_iter, ddim_steps):
    image_gen_config_df["final_img_path"] = final_img_path
    image_gen_config_df["seed_var"] = int(seed_var)
    image_gen_config_df["ddim_eta_var"] = float(ddim_eta_var)
    image_gen_config_df["scale_var"] = float(scale_var)
    image_gen_config_df["prompt_name"] = prompt_name
    image_gen_config_df["n_samples"] = int(n_samples)
    image_gen_config_df["n_iter"] = int(n_iter)
    image_gen_config_df["ddim_steps"] = int(ddim_steps)

    # update json file
    update_JSON()

    # create directories if necessary
    create_data_dirs()

    return image_gen_config_df["final_img_path"]

def image_generation_button():
    prompt = image_gen_config_df['prompt_name'].replace('_', ' ')
    image_gen_cmd = f"python scripts/stable_txt2img.py --seed {image_gen_config_df['seed_var']} --ddim_eta {image_gen_config_df['ddim_eta_var']} --n_samples {image_gen_config_df['n_samples']} --n_iter {image_gen_config_df['n_iter']} --scale {image_gen_config_df['scale_var']} --ddim_steps {image_gen_config_df['ddim_steps']} --ckpt {model_config_df['model_name']} --prompt \'{prompt}\' --outdir {image_gen_config_df['final_img_path']}"

    print("============================== IMAGE GENERATION TEST ==============================")
    print(image_gen_cmd)
    print("============================== --------------------- ==============================")

    if ("regularizer_var" in image_gen_config_df and image_gen_config_df["regularizer_var"] == 1):
        if "seed_var" in image_gen_config_df and "ddim_eta_var" in image_gen_config_df and "n_samples" in image_gen_config_df and "n_iter" in image_gen_config_df and "scale_var" in image_gen_config_df and "ddim_steps" in image_gen_config_df and "model_name" in model_config_df and "prompt_name" in image_gen_config_df and "final_img_path" in image_gen_config_df:
            return sub.run(image_gen_cmd.split(" "), stdout=sub.PIPE).stdout.decode('utf-8')
    else:
        return "Auto Regularization Method NOT Selected. Please Select the Correct Option to Generate (Regularization Images)\n" \
               "Please make sure to SAVE your settings before trying to Generate and/or Train."

def train_save_button(max_training_steps, batch_size):
    train_config_df["max_training_steps"] = int(max_training_steps)
    train_config_df["batch_size"] = int(batch_size)

    # update json file
    update_JSON()

    # create directories if necessary
    create_data_dirs()

def train_button():
    # train the model
    prompt = dataset_config_df['class_word'].replace('_', ' ')
    train_cmd = f"python main.py --base {dataset_config_df['config_path']} -t --actual_resume {model_config_df['model_name']} --reg_data_root {image_gen_config_df['final_img_path']} -n {dataset_config_df['project_name']} --gpus {system_config_df['gpu_used_var']}, --data_root {dataset_config_df['dataset_path']} --max_training_steps {train_config_df['max_training_steps']} --class_word {prompt} --no-test"

    print("============================== TRAINING COMMAND TEST ==============================")
    print(train_cmd)
    print("============================== --------------------- ==============================")

    if ("regularizer_var" in image_gen_config_df and image_gen_config_df["regularizer_var"] == 1):
        if 'config_path' in dataset_config_df and 'model_name' in model_config_df and 'final_img_path' in image_gen_config_df and 'project_name' in dataset_config_df and 'gpu_used_var' in system_config_df and 'dataset_path' in dataset_config_df and 'max_training_steps' in train_config_df and prompt:
            return sub.run(train_cmd.split(" "), stdout=sub.PIPE).stdout.decode('utf-8')
    else:
        return "Please make sure to SAVE your settings before trying to Generate and/or Train."

with gr.Blocks() as demo:
    with gr.Tab("Model & Data Configuration"):
        config_save_var = gr.Button(value="Apply & Save Settings", variant='primary')
        gr.Markdown(
        """
        ### Make sure a stable diffusion model with the (.ckpt) extension has been downloaded
        ### Please move the downloaded model into "this" repository folder
        """)

        print("ckpt_files", ckpt_files)

        with gr.Row():
            if "model_name" in model_config_df:
                model_var = gr.inputs.Radio(ckpt_files, type="value", default=model_config_df["model_name"], label='Select Model')
            else:
                model_var = gr.inputs.Radio(ckpt_files, type="value", label='Select Model')

        if "project_name" in dataset_config_df:
            project_name = gr.Textbox(lines=1, interactive=True, label='Class Target Name', value=dataset_config_df["project_name"])
        else:
            project_name = gr.Textbox(lines=1, interactive=True, label='Class Target Name')
        if "class_word" in dataset_config_df:
            class_word = gr.Textbox(lines=1, interactive=True, label='Regularization Class Word or Prompt', value=dataset_config_df["class_word"])
        else:
            class_word = gr.Textbox(lines=1, interactive=True, label='Regularization Class Word or Prompt')
        if "config_path" in dataset_config_df:
            config_path = gr.Textbox(lines=1, interactive=True, label='Path to Model YAML Config', value=dataset_config_df["config_path"])
        else:
            config_path = gr.Textbox(lines=1, interactive=True, label='Path to Model YAML Config')
        if "dataset_path" in dataset_config_df:
            dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Class Target Dataset', value=dataset_config_df["dataset_path"])
        else:
            dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Class Target Dataset')
        if "reg_dataset_path" in dataset_config_df:
            reg_dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Regularization Dataset', value=dataset_config_df["reg_dataset_path"])
        else:
            reg_dataset_path = gr.Textbox(lines=1, interactive=True, label='Path to Regularization Dataset')

        with gr.Row():
            import torch
            if not "gpu_used_var" in system_config_df:
                system_config_df["gpu_used_var"] = [i for i in range(0, torch.cuda.device_count())][0] # EXPECT THIS TO CHANGE IN THE FUTURE
            temp_text = [f"gpu: {i}" for i in range(0, torch.cuda.device_count())]
            gpu_used_var = gr.inputs.Radio(temp_text, default=temp_text[(system_config_df["gpu_used_var"])], label='Select GPU')

    model_var.change(fn=model_choice, inputs=[model_var], outputs=[])
    config_save_var.click(fn=model_config_save_button,
                          inputs=[model_var,
                                  gpu_used_var,
                                  project_name,
                                  class_word,
                                  config_path,
                                  dataset_path,
                                  reg_dataset_path
                                  ],
                          outputs=[]
                          )

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
                                            value=dataset_config_df["reg_dataset_path"])
            elif "final_img_path" in image_gen_config_df:
                final_img_path = gr.Textbox(lines=1, interactive=True, label='Regularization Data Path',
                                            value=image_gen_config_df["final_img_path"])

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
        if "prompt_name" in image_gen_config_df:
            prompt_name = gr.Textbox(lines=1, interactive=True, label='Regularization Class Word or Prompt', value=image_gen_config_df["prompt_name"])
        else:
            prompt_name = gr.Textbox(lines=1, interactive=True, label='Regularization Class Word or Prompt')

        if "n_samples" in image_gen_config_df:
            n_samples = gr.Slider(minimum=0, maximum=200, step=1, label='Samples per Iteration', value=int(image_gen_config_df["n_samples"]))
        else:
            n_samples = gr.Slider(minimum=0, maximum=200, step=1, value=1, label='Samples per Iteration')
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
        image_gen_output = gr.Textbox(lines=16, interactive=False, label='Regularized Image Generation Logs')

    regularizer_var.change(fn=change_regularizer_view, inputs=[regularizer_var], outputs=[])
    regularizer_save_var.click(fn=image_gen_config_save_button, inputs=[final_img_path,
                                                                        seed_var,
                                                                        ddim_eta_var,
                                                                        scale_var,
                                                                        prompt_name,
                                                                        n_samples,
                                                                        n_iter,
                                                                        ddim_steps
                                                                        ], outputs=[final_img_path])
    generate_images_var.click(fn=image_generation_button, inputs=[], outputs=[image_gen_output], show_progress=True, scroll_to_output=True)

    with gr.Tab("Fine-Tuning Model"):
        fine_tine_save_var = gr.Button(value="Apply & Save Settings", variant='primary')

        if "max_training_steps" in train_config_df:
            max_training_steps = gr.Slider(minimum=0, maximum=20000, step=20, label='Max Training Steps', value=int(train_config_df["max_training_steps"]))
        else:
            max_training_steps = gr.Slider(minimum=0, maximum=20000, step=20, label='Max Training Steps', value=2000)
        if "batch_size" in train_config_df:
            batch_size = gr.Slider(minimum=0, maximum=64, step=1, label='Batch Size', value=int(train_config_df["batch_size"]))
        else:
            batch_size = gr.Slider(minimum=0, maximum=64, step=1, label='Batch Size', value=1)

        with gr.Row():
            train_out_var = gr.Button(value="Generate", variant='secondary')
        train_output = gr.Textbox(lines=30, interactive=False, label='Training Logs')

    fine_tine_save_var.click(fn=train_save_button, inputs=[max_training_steps,
                                                                        batch_size
                                                                        ], outputs=[])
    train_out_var.click(fn=train_button, inputs=[], outputs=[train_output], show_progress=True, scroll_to_output=True)

if __name__ == "__main__":
    demo.launch()

