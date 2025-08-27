from .settings import *
import os
import time

'''
params setting
'''
settings = JsonConfig()
settings.load_json_file("train_settings.json")

with_diffusion = settings.with_diffusion
only_decoder = settings.only_decoder

project_name = settings.project_name
dataset_path = settings.dataset_path
dataset_train = settings.dataset_train
epoch_number = settings.epoch_number
batch_size = settings.batch_size
train_continue = settings.train_continue
train_continue_path = settings.train_continue_path
train_continue_epoch = settings.train_continue_epoch
save_images_number = settings.save_images_number
lr = settings.lr
SAVE_EVERY_N_STEPS = settings.SAVE_EVERY_N_STEPS
VALIDATE_EVERY_N_STEPS = settings.VALIDATE_EVERY_N_STEPS
train_continue_step = settings.train_continue_step
H, W, message_length = settings.H, settings.W, settings.message_length,
noise_layers = settings.noise_layers
experiment_name = settings.experiment_name

'''
file preparing
'''
if train_continue:
	full_project_name = experiment_name
	result_folder = "runs/" + full_project_name + "/" + dataset_train + "/"
else:
	full_project_name = project_name + '_' + str(H) +"_m" + str(message_length)
	#for noise in noise_layers:
	#	full_project_name += "_" + noise

	result_folder = "runs/" + time.strftime(full_project_name + "__%Y_%m_%d__%H_%M_%S", time.localtime()) + "/" + dataset_train + "/"
	if not os.path.exists(result_folder): os.makedirs(result_folder)
	if not os.path.exists(result_folder + "samples/"): os.makedirs(result_folder + "samples/")
	if not os.path.exists(result_folder + "model/"): os.makedirs(result_folder + "model/")

	with open(result_folder + "/train_params.txt", "w") as file:
		content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
															time.localtime()) + "-----------------------\n"

		for item in settings.get_items():
			content += item[0] + " = " + str(item[1]) + "\n"

		print(content)
		file.write(content)

	with open(result_folder + "/train_log.txt", "w") as file:
		content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
															time.localtime()) + "-----------------------\n"
		file.write(content)
	with open(result_folder + "/val_log.txt", "w") as file:
		content = "-----------------------" + time.strftime("Date: %Y/%m/%d %H:%M:%S",
															time.localtime()) + "-----------------------\n"
		file.write(content)
