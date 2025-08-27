from torch.utils.data import DataLoader
from utils import *
from tqdm import tqdm
from network.Network import *
import torch
import os

from utils.load_train_setting import *

'''
train
'''

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")
#torch.backends.cudnn.enabled = False
#device = torch.device("cpu")
print("Device: ", device)

print("Creating network...")
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, with_diffusion, only_decoder)

print("Creating datasets...")
train_dataset = MBRSDataset(os.path.join(dataset_path, dataset_train, "train", "real"), H, W)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6, pin_memory=True)

val_dataset = MBRSDataset(os.path.join(dataset_path, dataset_train, "val", "real"), H, W)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=6, pin_memory=True)

if train_continue:
	print("Continuing training from epoch", train_continue_epoch, "step", train_continue_step)
	EC_path = result_folder + "/model/" + f"EC_epoch{train_continue_epoch}_step{train_continue_step}.pth"
	D_path = result_folder + "/model/" + f"D_epoch{train_continue_epoch}_step{train_continue_step}.pth"
	network.load_model(EC_path, D_path)

print("\nStart training : \n\n")

path_model = result_folder + "model/"
global_step = 0
for epoch in tqdm(range(epoch_number), desc="Epochs"):
	epoch += train_continue_epoch if train_continue else 0
	global_step += train_continue_step if train_continue else 0

	running_result = {
		"error_rate": 0.0,
		"psnr": 0.0,
		"ssim": 0.0,
		"g_loss": 0.0,
		"g_loss_on_discriminator": 0.0,
		"g_loss_on_encoder": 0.0,
		"g_loss_on_decoder": 0.0,
		"d_cover_loss": 0.0,
		"d_encoded_loss": 0.0
	}

	start_time = time.time()

	'''
	train
	'''
	num = 0
	interval_start_time = time.time()
	for step_idx, images, in tqdm(enumerate(train_dataloader), desc=f"Epoch {epoch} Steps", leave=False):
		global_step += 1

		image = images.to(device)
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

		result = network.train(image, message) if not only_decoder else network.train_only_decoder(image, message)

		for key in result:
			running_result[key] += float(result[key])

		num += 1
		if (global_step + 1) % SAVE_EVERY_N_STEPS == 0:
			path_encoder_decoder = os.path.join(path_model, f"EC_epoch{epoch}_step{global_step}.pth")
			path_discriminator = os.path.join(path_model, f"D_epoch{epoch}_step{global_step}.pth")

			tqdm.write(f"\n--- Evaluating and Storing model at global step: {global_step} (Epoch {epoch}, Step internal {step_idx}) ---")
			network.save_model(path_encoder_decoder, path_discriminator)

			interval_time_taken = int(time.time() - interval_start_time)
			content = f"Epoch {epoch} : Global Step {global_step} : Internal step {step_idx} : {interval_time_taken}s\n"
			for key in running_result:
				content += key + "=" + str(running_result[key] / num) + ","
			content += "\n"

			with open(result_folder + "/train_log.txt", "a") as file:
				file.write(content)
			tqdm.write(content)

			running_result = {
				"error_rate": 0.0,
				"psnr": 0.0,
				"ssim": 0.0,
				"g_loss": 0.0,
				"g_loss_on_discriminator": 0.0,
				"g_loss_on_encoder": 0.0,
				"g_loss_on_decoder": 0.0,
				"d_cover_loss": 0.0,
				"d_encoded_loss": 0.0
			}

		if (global_step + 1) % VALIDATE_EVERY_N_STEPS == 0:

			'''
			validation
			'''

			val_result = {
				"error_rate": 0.0,
				"psnr": 0.0,
				"ssim": 0.0,
				"g_loss": 0.0,
				"g_loss_on_discriminator": 0.0,
				"g_loss_on_encoder": 0.0,
				"g_loss_on_decoder": 0.0,
				"d_cover_loss": 0.0,
				"d_encoded_loss": 0.0
			}

			start_time = time.time()

			saved_iterations = np.random.choice(np.arange(len(val_dataloader)), size=save_images_number, replace=False)
			saved_all = None

			num = 0
			for i, images in tqdm(enumerate(val_dataloader), desc=f"Val: Epoch {epoch} Steps", leave=False):
				image = images.to(device)
				message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

				result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message)

				for key in result:
					val_result[key] += float(result[key])

				num += 1

				if i in saved_iterations:
					if saved_all is None:
						saved_all = get_random_images(image, encoded_images, noised_images)
					else:
						saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

			save_images(saved_all, str(epoch)+'_'+str(global_step), result_folder + "samples/", resize_to=(W, H))

			'''
			validation results
			'''
			content = f"Epoch {epoch} : Global Step {global_step} : Internal step {step_idx} : {interval_time_taken}s\n"
			for key in val_result:
				content += key + "=" + str(val_result[key] / num) + ","
			content += "\n"

			with open(result_folder + "/val_log.txt", "a") as file:
				file.write(content)
			tqdm.write(content)

			val_result = {
				"error_rate": 0.0,
				"psnr": 0.0,
				"ssim": 0.0,
				"g_loss": 0.0,
				"g_loss_on_discriminator": 0.0,
				"g_loss_on_encoder": 0.0,
				"g_loss_on_decoder": 0.0,
				"d_cover_loss": 0.0,
				"d_encoded_loss": 0.0
			}

	'''
	save model
	'''
	path_encoder_decoder = os.path.join(path_model, f"EC_epoch{epoch}_global_step{step_idx}.pth")
	path_discriminator = os.path.join(path_model, f"D_epoch{epoch}_global_step{step_idx}.pth")
	network.save_model(path_encoder_decoder, path_discriminator)
