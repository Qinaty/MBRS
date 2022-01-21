from . import *
from .noise_layers import *


class Noise(nn.Module):

	def __init__(self, layers):
		# layers Combined([JpegMask(50),Jpeg(50),Identity()])
		super(Noise, self).__init__()
		# print(len(layers))
		for i in range(len(layers)):
			# layers[i] = Combined(layers[i])
			layers[i] = eval(layers[i])
		self.noise = nn.Sequential(*layers)

	def forward(self, image_and_cover):
		noised_image = self.noise(image_and_cover)
		return noised_image
