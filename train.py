from utils import *
from network.Network import *

from utils.load_train_setting import *

import os

# Modify: for grad_cam
from torchvision import transforms
from gradcam import GradCAM
import torchvision.models as models

'''
train
'''

# Modify: Choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = False
if mask_type == "attention":
	attention = True
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, attention, with_diffusion, only_decoder)

dataloader = Dataloader(batch_size, dataset_path, H=H, W=W)
train_dataloader = dataloader.load_train_data()
val_dataloader = dataloader.load_val_data()

# Modify: grad_cam
vgg16 = models.vgg16(pretrained=True).to(device)
vgg16.eval()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
conf = dict(model_type='vgg', arch=vgg16, layer_name='features_29')
camera = GradCAM.from_config(**conf)

if train_continue:
	EC_path = "results/" + train_continue_path + "/models/EC_" + str(train_continue_epoch) + ".pth"
	D_path = "results/" + train_continue_path + "/models/D_" + str(train_continue_epoch) + ".pth"
	network.load_model(EC_path, D_path)

print("\nStart training : \n\n")

for epoch in range(epoch_number):

	epoch += train_continue_epoch if train_continue else 0

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

	# Modify: record best
	# best_result = running_result

	start_time = time.time()

	'''
	train
	'''
	num = 0
	for _, images, in enumerate(train_dataloader):
		image = images.to(device)
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

		# Modify
		if mask_type == "opt":
			mask = torch.empty_like(image)[:, 0:1, :, :].normal_(mean=-2, std=1)
			# print(mask.shape[1])
			mask = network.train_mask(image, message, mask)
			mask = mask.to(device)
		elif mask_type == "uniform":
			mask = torch.ones_like(image)[:, 0:1, :, :].to(device)
			mask.requires_grad = False
		elif mask_type == "grad_cam":
			alpha = 0.5
			# image_ = None
			# get C*H*W
			# for pic in image:
			# 	pic = normalize(pic.squeeze()).unsqueeze(0)
			# 	print(type(pic))
			# 	if image_ is None:
			# 		image_ = pic
			# 	else:
			# 		image_ = torch.cat([image_, pic], dim=0)
			# print("!!", type(image_))

			mask, _ = camera(normalize(image.squeeze()).unsqueeze(0))
			# print(mask.shape)
			mask = alpha * mask.to(device)
		else:
			mask = None

		if attention:
			result, mask = network.train_attention(image, message)
			# print(mask.shape)
		else:
			result = network.train(image, message, mask) if not only_decoder else network.train_only_decoder(image, message)

		for key in result:
			running_result[key] += float(result[key])

		num += 1

	'''
	train results
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
	for key in running_result:
		content += key + "=" + str(running_result[key] / num) + ","
	content += "\n"

	with open(result_folder + "/train_log.txt", "a") as file:
		file.write(content)
	print(content)

	# '''
	# validation
	# '''

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
	for i, images in enumerate(val_dataloader):
		image = images.to(device)
		message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

		if attention:
			result, (images, encoded_images, noised_images, messages, decoded_messages, mask) = \
				network.validation_attention(image, message)
		else:
			if mask_type == "opt":
				mask = torch.empty_like(image)[:, 0:1, :, :].normal_(mean=-2, std=1)
				mask = network.train_mask(image, message, mask)
				mask = mask.to(device)
			elif mask_type == "uniform":
				mask = torch.ones_like(image)[:, 0:1, :, :].to(device)
				mask.requires_grad = False
			elif mask_type == "grad_cam":
				alpha = 0.5
				mask, _ = camera(normalize(image.squeeze()).unsqueeze(0))
				# print(mask.shape)
				mask = alpha * mask.to(device)
			else:
				mask = None

			result, (images, encoded_images, noised_images, messages, decoded_messages) = network.validation(image, message, mask)

		for key in result:
			val_result[key] += float(result[key])

		num += 1

		# Modify: save mask
		if attention:
			if i in saved_iterations:
				if saved_all is None:
					saved_all = get_random_images_mask(image, encoded_images, noised_images, mask)
				else:
					saved_all = concatenate_images_mask(saved_all, image, encoded_images, noised_images, mask)

		elif mask_type == "opt":
			if i in saved_iterations:
				if saved_all is None:
					saved_all = get_random_images(image, encoded_images, mask)
				else:
					saved_all = concatenate_images(saved_all, image, encoded_images, mask)
		else:
			if i in saved_iterations:
				if saved_all is None:
					saved_all = get_random_images(image, encoded_images, noised_images)
				else:
					saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)

	if attention:
		save_images_mask(saved_all, epoch, result_folder + "images/", resize_to=(W, H))
	else:
		save_images(saved_all, epoch, result_folder + "images/", resize_to=(W, H))

	'''
	validation results
	'''
	content = "Epoch " + str(epoch) + " : " + str(int(time.time() - start_time)) + "\n"
	for key in val_result:
		content += key + "=" + str(val_result[key] / num) + ","
	content += "\n"

	with open(result_folder + "/val_log.txt", "a") as file:
		file.write(content)
	print(content)

	'''
	save model
	'''
	path_model = result_folder + "models/"
	path_encoder_decoder = path_model + "EC_" + str(epoch) + ".pth"
	path_discriminator = path_model + "D_" + str(epoch) + ".pth"
	network.save_model(path_encoder_decoder, path_discriminator)
