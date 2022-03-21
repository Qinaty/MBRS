from . import *
from .Encoder_MP import Encoder_MP, Encoder_MP_Diffusion, Encoder_Mask, Encoder_Mask_Loss
from .Decoder import Decoder, Decoder_Diffusion
from .Noise import Noise


class EncoderDecoder(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers, grad_cam=False):
		super(EncoderDecoder, self).__init__()
		# print("Encoder_MP_Decoder: ", grad_cam)
		self.encoder = Encoder_MP(H, W, message_length, grad_cam=grad_cam)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder(H, W, message_length)

	def forward(self, image, message, mask):
		encoded_image = self.encoder(image, message, mask)
		noised_image = self.noise([encoded_image, image])
		decoded_message = self.decoder(noised_image)
		return encoded_image, noised_image, decoded_message


class EncoderDecoder_Diffusion(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder_Diffusion, self).__init__()
		self.encoder = Encoder_MP_Diffusion(H, W, message_length)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder_Diffusion(H, W, message_length)

	def forward(self, image, message):
		encoded_image = self.encoder(image, message)
		noised_image = self.noise([encoded_image, image])
		decoded_message = self.decoder(noised_image)

		return encoded_image, noised_image, decoded_message


# Modify: NLAMask
class EncoderDecoder_Mask(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder_Mask, self).__init__()
		self.encoder = Encoder_Mask(H, W, message_length)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder(H, W, message_length)

	def forward(self, image, message):
		encoded_image, mask = self.encoder(image, message)
		noised_image = self.noise([encoded_image, image])
		decoded_message = self.decoder(noised_image)
		return encoded_image, noised_image, decoded_message, mask


# Modify: Add loss
class EncoderDecoder_Mask_Loss(nn.Module):
	'''
	A Sequential of Encoder_MP-Noise-Decoder
	'''

	def __init__(self, H, W, message_length, noise_layers):
		super(EncoderDecoder_Mask_Loss, self).__init__()
		self.encoder = Encoder_Mask_Loss(H, W, message_length)
		self.noise = Noise(noise_layers)
		self.decoder = Decoder(H, W, message_length)

	def forward(self, image, message):
		encoded_image, mask, original_features, encoded_features = self.encoder(image, message)
		noised_image = self.noise([encoded_image, image])
		decoded_message = self.decoder(noised_image)
		return encoded_image, noised_image, decoded_message, mask, original_features, encoded_features