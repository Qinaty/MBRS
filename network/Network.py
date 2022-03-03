from .Encoder_MP_Decoder import *
from .Discriminator import Discriminator
# from utils.load_train_setting import mask_iter


class Network:

	def __init__(self, H, W, message_length, noise_layers, device, batch_size, lr, attention, with_diffusion=False,
				 only_decoder=False):
		# device
		self.device = device

		# network
		if attention:
			self.encoder_decoder = EncoderDecoder_Mask(H, W, message_length, noise_layers).to(device)
		else:
			if not with_diffusion:
				self.encoder_decoder = EncoderDecoder(H, W, message_length, noise_layers).to(device)
			else:
				self.encoder_decoder = EncoderDecoder_Diffusion(H, W, message_length, noise_layers).to(device)

		# self.encoder_decoder_mask = EncoderDecoder_Mask(H, W, message_length, noise_layers).to(device)

		self.discriminator = Discriminator().to(device)

		self.encoder_decoder = torch.nn.DataParallel(self.encoder_decoder)  # DataParallel mini-batch送到不同device训练
		self.discriminator = torch.nn.DataParallel(self.discriminator)
		# Modify
		# self.encoder_decoder_mask = torch.nn.DataParallel(self.encoder_decoder_mask)

		if only_decoder:
			for p in self.encoder_decoder.module.encoder.parameters():
				p.requires_grad = False

		# mark "cover" as 1, "encoded" as 0
		self.label_cover = torch.full((batch_size, 1), 1, dtype=torch.float, device=device)
		self.label_encoded = torch.full((batch_size, 1), 0, dtype=torch.float, device=device)

		# optimizer
		print(lr)
		self.opt_encoder_decoder = torch.optim.Adam(
			filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)
		self.opt_discriminator = torch.optim.Adam(self.discriminator.parameters(), lr=lr)
		# # Modify
		# self.opt_encoder_decoder_mask = torch.optim.Adam(
		# 	filter(lambda p: p.requires_grad, self.encoder_decoder.parameters()), lr=lr)

		# loss function
		self.criterion_BCE = nn.BCEWithLogitsLoss().to(device)
		self.criterion_MSE = nn.MSELoss().to(device)
		# Modify: 计算结果shape保持不变
		self.criterion_BMSE = nn.MSELoss(reduction='none').to(device)

		# weight of encoder-decoder loss
		self.discriminator_weight = 0.0001
		self.encoder_weight = 1
		self.mask_weight = 1
		self.decoder_weight = 10
		self.grad_clip = 1e1  # Modify: opt_mask grad_clip

	# Modify: train_with_mask
	# def train_mask(self, images: torch.Tensor, messages: torch.Tensor, mask: torch.Tensor):
	# 	self.encoder_decoder.eval()
	# 	self.discriminator.eval()
	#
	# 	mask_wograd = mask
	# 	mask_wograd.requires_grad = False
	# 	# print("best_mask", mask_wograd)
	#
	# 	mask = mask.clone().detach().to(self.device)
	# 	mask.requires_grad_()
	# 	opt_mask = torch.optim.LBFGS([mask], max_iter=1)
	# 	# print("mask", mask)
	# 	# print("opt_mask", opt_mask)
	#
	# 	mask_paras = {
	# 		'image': images,
	# 		'message': messages,
	# 		'mask': mask,
	# 		'best_loss': float('inf'),
	# 		'best_mask': mask_wograd,
	# 		'encoded_image': None,
	# 		'optimizer': opt_mask,
	# 		'train_iter': mask_iter
	# 	}
	#
	# 	with torch.enable_grad():
	# 		for iter in range(1, mask_paras['train_iter']):
	# 			def closure():
	# 				# nonlocal - change outer variable
	# 				nonlocal mask_paras
	# 				images_ = mask_paras['image']
	# 				messages_ = mask_paras['message']
	# 				mask_ = mask_paras['mask']
	# 				# print("mask_", mask_)
	#
	# 				images_, messages_ = images_.to(self.device), messages_.to(self.device)
	# 				encoded_image, noised_image, decoded_message = self.encoder_decoder(images_, messages_)
	#
	# 				# GAN : target label for encoded image should be "cover"(0)
	# 				g_label_decoded = self.discriminator(encoded_image)
	# 				loss_on_discriminator = self.criterion_BCE(g_label_decoded,
	# 														self.label_cover[:g_label_decoded.shape[0]])
	#
	# 				# encoder
	# 				mse_loss = self.criterion_BMSE(encoded_image, images_)
	# 				mask_ex = mask_.expand_as(mse_loss)  # 对每个channel使用同样的mask
	# 				loss_on_encoder = abs(torch.mean(mask_ex * mse_loss))
	#
	# 				# decoder
	# 				g_loss_on_decoder = self.criterion_MSE(decoded_message, messages_)
	#
	# 				mask_loss = self.discriminator_weight * loss_on_discriminator + \
	# 							self.encoder_weight * loss_on_encoder + \
	# 							self.decoder_weight * g_loss_on_decoder
	#
	# 				mask_paras['optimizer'].zero_grad()
	# 				mask_loss.backward()
	#
	# 				if mask_loss < mask_paras['best_loss']:
	# 					mask_paras['best_loss'] = mask_loss
	# 					best_mask = mask_.clone().detach()
	# 					# print("best_mask", best_mask)
	# 					mask_paras['best_mask'] = best_mask
	# 					mask_paras['encoded_image'] = encoded_image
	#
	# 				torch.nn.utils.clip_grad_norm_(mask_, self.grad_clip)
	#
	# 				return mask_loss
	# 			# print("iter:", iter)
	# 			opt_mask.step(closure)
	#
	# 	return mask_paras['best_mask']

	def train(self, images: torch.Tensor, messages: torch.Tensor, mask):
		self.encoder_decoder.train()
		self.discriminator.train()

		with torch.enable_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

			'''
			train discriminator
			'''
			self.opt_discriminator.zero_grad()

			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
			d_cover_loss.backward()

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			d_encoded_loss.backward()

			self.opt_discriminator.step()  # optimizer.step()对应每一个mini-batch

			'''
			train encoder and decoder
			'''
			self.opt_encoder_decoder.zero_grad()

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			if mask == None:
				g_loss_on_encoder = self.criterion_MSE(encoded_images, images)
			# Modify: mask
			else:
				g_loss_on_encoder = self.criterion_BMSE(encoded_images, images)
				mask = mask.expand_as(g_loss_on_encoder)
				g_loss_on_encoder = abs(torch.mean(mask * g_loss_on_encoder))

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder

			g_loss.backward()
			self.opt_encoder_decoder.step()

			# psnr
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), images, window_size=5, reduction="mean")
			# ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}
		return result

	# Modify: train_with NLAMask
	def train_attention(self, images, messages):
		self.encoder_decoder.train()
		self.discriminator.train()

		with torch.enable_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages, mask = self.encoder_decoder(images, messages)
			mask = torch.sum(mask, 1, keepdim=True)

			'''
			train discriminator
			'''
			self.opt_discriminator.zero_grad()

			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])
			d_cover_loss.backward()

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])
			d_encoded_loss.backward()

			self.opt_discriminator.step()  # optimizer.step()对应每一个mini-batch

			'''
			train encoder and decoder
			'''
			self.opt_encoder_decoder.zero_grad()

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			# Modify: mask
			g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

			# g_loss_on_encoder = self.criterion_BMSE(encoded_images, images)
			# g_loss_on_encoder = abs(torch.mean(mask * g_loss_on_encoder))

			# Modify: mask loss
			residual_images = (encoded_images - images + 1) / 2
			mask1 = mask.expand_as(residual_images)
			g_loss_on_mask = self.criterion_MSE(residual_images, mask1)

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder + self.mask_weight * g_loss_on_mask

			g_loss.backward()
			self.opt_encoder_decoder.step()

			# psnr
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), images, window_size=5, reduction="mean")
			# ssim = 1 - 2 * kornia.losses.ssim_loss(encoded_images.detach(), images, window_size=5, reduction="mean")

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}
		return result, mask

	# def train_only_decoder(self, images: torch.Tensor, messages: torch.Tensor):
	# 	self.encoder_decoder.train()
	#
	# 	with torch.enable_grad():
	# 		# use device to compute
	# 		images, messages = images.to(self.device), messages.to(self.device)
	# 		encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)
	#
	# 		'''
	# 		train encoder and decoder
	# 		'''
	# 		self.opt_encoder_decoder.zero_grad()
	#
	# 		# RESULT : the decoded message should be similar to the raw message
	# 		g_loss = self.criterion_MSE(decoded_messages, messages)
	#
	# 		g_loss.backward()
	# 		self.opt_encoder_decoder.step()
	#
	# 		# psnr
	# 		psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)
	#
	# 		# ssim
	# 		ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), images, window_size=5, reduction="mean")
	#
	# 	'''
	# 	decoded message error rate
	# 	'''
	# 	error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)
	#
	# 	result = {
	# 		"error_rate": error_rate,
	# 		"psnr": psnr,
	# 		"ssim": ssim,
	# 		"g_loss": g_loss,
	# 		"g_loss_on_discriminator": 0.,
	# 		"g_loss_on_encoder": 0.,
	# 		"g_loss_on_decoder": 0.,
	# 		"d_cover_loss": 0.,
	# 		"d_encoded_loss": 0.
	# 	}
	# 	return result

	def validation(self, images: torch.Tensor, messages: torch.Tensor, mask):
		self.encoder_decoder.eval()
		self.discriminator.eval()

		with torch.no_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

			'''
			validate discriminator
			'''
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

			'''
			validate encoder and decoder
			'''

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			if mask == None:
				g_loss_on_encoder = self.criterion_MSE(encoded_images, images)
			# Modify: mask
			else:
				g_loss_on_encoder = self.criterion_BMSE(encoded_images, images)
				mask = mask.expand_as(g_loss_on_encoder)
				g_loss_on_encoder = abs(torch.mean(mask * g_loss_on_encoder))

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder

			# psnr
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), images, window_size=5, reduction="mean")

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}

		return result, (images, encoded_images, noised_images, messages, decoded_messages)

	# Modify
	def validation_attention(self, images: torch.Tensor, messages: torch.Tensor):
		self.encoder_decoder.eval()
		self.discriminator.eval()

		with torch.no_grad():
			# use device to compute
			images, messages = images.to(self.device), messages.to(self.device)
			encoded_images, noised_images, decoded_messages, mask = self.encoder_decoder(images, messages)
			mask = torch.sum(mask, 1, keepdim=True)

			'''
			validate discriminator
			'''
			# RAW : target label for image should be "cover"(1)
			d_label_cover = self.discriminator(images)
			d_cover_loss = self.criterion_BCE(d_label_cover, self.label_cover[:d_label_cover.shape[0]])

			# GAN : target label for encoded image should be "encoded"(0)
			d_label_encoded = self.discriminator(encoded_images.detach())
			d_encoded_loss = self.criterion_BCE(d_label_encoded, self.label_encoded[:d_label_encoded.shape[0]])

			'''
			validate encoder and decoder
			'''

			# GAN : target label for encoded image should be "cover"(0)
			g_label_decoded = self.discriminator(encoded_images)
			g_loss_on_discriminator = self.criterion_BCE(g_label_decoded, self.label_cover[:g_label_decoded.shape[0]])

			# RAW : the encoded image should be similar to cover image
			g_loss_on_encoder = self.criterion_MSE(encoded_images, images)

			# RESULT : the decoded message should be similar to the raw message
			g_loss_on_decoder = self.criterion_MSE(decoded_messages, messages)

			# full loss
			g_loss = self.discriminator_weight * g_loss_on_discriminator + self.encoder_weight * g_loss_on_encoder + \
					 self.decoder_weight * g_loss_on_decoder

			# psnr
			psnr = kornia.losses.psnr_loss(encoded_images.detach(), images, 2)

			# ssim
			ssim = 1 - 2 * kornia.losses.ssim(encoded_images.detach(), images, window_size=5, reduction="mean")

		'''
		decoded message error rate
		'''
		error_rate = self.decoded_message_error_rate_batch(messages, decoded_messages)

		result = {
			"error_rate": error_rate,
			"psnr": psnr,
			"ssim": ssim,
			"g_loss": g_loss,
			"g_loss_on_discriminator": g_loss_on_discriminator,
			"g_loss_on_encoder": g_loss_on_encoder,
			"g_loss_on_decoder": g_loss_on_decoder,
			"d_cover_loss": d_cover_loss,
			"d_encoded_loss": d_encoded_loss
		}

		return result, (images, encoded_images, noised_images, messages, decoded_messages, mask)

	def decoded_message_error_rate(self, message, decoded_message):
		length = message.shape[0]

		message = message.gt(0.5)
		decoded_message = decoded_message.gt(0.5)
		error_rate = float(sum(message != decoded_message)) / length
		return error_rate

	def decoded_message_error_rate_batch(self, messages, decoded_messages):
		error_rate = 0.0
		batch_size = len(messages)
		for i in range(batch_size):
			error_rate += self.decoded_message_error_rate(messages[i], decoded_messages[i])
		error_rate /= batch_size
		return error_rate

	def save_model(self, path_encoder_decoder: str, path_discriminator: str):
		torch.save(self.encoder_decoder.module.state_dict(), path_encoder_decoder)
		torch.save(self.discriminator.module.state_dict(), path_discriminator)

	def load_model(self, path_encoder_decoder: str, path_discriminator: str):
		self.load_model_ed(path_encoder_decoder)
		self.load_model_dis(path_discriminator)

	def load_model_ed(self, path_encoder_decoder: str):
		self.encoder_decoder.module.load_state_dict(torch.load(path_encoder_decoder))

	def load_model_dis(self, path_discriminator: str):
		self.discriminator.module.load_state_dict(torch.load(path_discriminator))

	# Modify:
	def get_features(self, images: torch.Tensor):
		with torch.no_grad():
			features = self.encoder_decoder.encoder
