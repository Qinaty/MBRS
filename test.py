from utils import *
from network.Network import *

from utils.load_test_setting import *

# Modify: for grad_cam
from torchvision import transforms
from gradcam import GradCAM
import torchvision.models as models

'''
test
'''

# Modify: Choose GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

attention = False
addloss = False
grad_cam = False
mask = None

if mask_type == "attention":
	attention = True
if mask_type == "attention_plus":
	attention = True
	addloss = True
if mask_type == "gradcam":
	grad_cam = True
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, attention,
					addloss, grad_cam=grad_cam, with_diffusion=with_diffusion)

EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
network.load_model_ed(EC_path)

dataloader = Dataloader(batch_size, dataset_path, H=H, W=W)
test_dataloader = dataloader.load_test_data()

# Mask-D: grad_cam
if grad_cam:
	vgg16 = models.vgg16(pretrained=True).to(device)
	vgg16.eval()
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	conf = dict(model_type='vgg', arch=vgg16, layer_name='features_29')
	camera = GradCAM.from_config(**conf)
	alpha = 0.1

print("\nStart Testing : \n\n")

test_result = {
	"error_rate": 0.0,
	"psnr": 0.0,
	"ssim": 0.0
}

start_time = time.time()

# saved_iterations = np.random.choice(np.arange(len(test_dataloader)), size=save_images_number, replace=False)
saved_iterations = [0, 1, 2, 3]
saved_all = None
# print(saved_iterations)

num = 0
for i, images in enumerate(test_dataloader):
	image = images.to(device)
	message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)

	'''
	test
	'''
	network.encoder_decoder.eval()
	network.discriminator.eval()

	if grad_cam:
		tmp_images = images.to(device)
		mask, _ = camera(normalize(tmp_images.squeeze(0)).unsqueeze(0))
		mask = alpha * mask
		mask = torch.sum(mask, 1, keepdim=True)

	with torch.no_grad():

		# if mask_type == "opt":
		# else:
		# 	mask = None
		if attention:
			if addloss:
				encoded_images, mask, _, _ = network.encoder_decoder.module.encoder(image, message)
			else:
				encoded_images, mask = network.encoder_decoder.module.encoder(image, message)
		else:
			encoded_images = network.encoder_decoder.module.encoder(image, message, mask)
		encoded_images = image + (encoded_images - image) * strength_factor
		noised_images = network.encoder_decoder.module.noise([encoded_images, image])

		decoded_messages = network.encoder_decoder.module.decoder(noised_images)

		# psnr
		psnr = kornia.losses.psnr_loss(encoded_images.detach(), image, 2).item()

		# ssim
		ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), image, window_size=5, reduction="mean").item()

	'''
	decoded message error rate
	'''
	error_rate = network.decoded_message_error_rate_batch(message, decoded_messages)

	result = {
		"error_rate": error_rate,
		"psnr": psnr,
		"ssim": ssim,
	}

	for key in result:
		test_result[key] += float(result[key])
	# print(mask.max())
	# print(mask.min())

	num += 1

	if attention:
		if i in saved_iterations:
			if saved_all is None:
				saved_all = get_random_images_mask(image, encoded_images, noised_images, mask)
			else:
				saved_all = concatenate_images_mask(saved_all, image, encoded_images, noised_images, mask)
	else:
		if i in saved_iterations:
			if saved_all is None:
				saved_all = get_random_images(image, encoded_images, noised_images)
			else:
				saved_all = concatenate_images(saved_all, image, encoded_images, noised_images)


	'''
	test results
	'''
	content = "Image " + str(i) + " : \n"
	for key in test_result:
		content += key + "=" + str(result[key]) + ","
	content += "\n"

	with open(test_log, "a") as file:
		file.write(content)

	print(content)

'''
test results
'''
content = "Average : \n"
for key in test_result:
	content += key + "=" + str(test_result[key] / num) + ","
content += "\n"

with open(test_log, "a") as file:
	file.write(content)

print(content)
if attention:
	save_images_mask(saved_all, "test_mask", result_folder + "images/", resize_to=(W, H))
else:
	save_images(saved_all, "test", result_folder + "images/", resize_to=(W, H))



