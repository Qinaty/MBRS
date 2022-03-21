from . import *


class Encoder_MP(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64, grad_cam=False):
		super(Encoder_MP, self).__init__()
		self.H = H
		self.W = W

		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))  # n
		message_se_blocks = max(blocks - message_convT_blocks, 1)

		self.image_pre_layer = ConvBNRelu(3, channels)  # 归一化
		self.image_first_layer = SENet(channels, channels, blocks=blocks)

		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),  # 依据channel number扩展
			SENet(channels, channels, blocks=message_se_blocks),  # SE特征提取
		)

		self.message_first_layer = SENet(channels, channels, blocks=blocks)  # image feature学习

		# print("Encoder_MP: ", grad_cam)
		if grad_cam:
			self.after_concat_layer = ConvBNRelu(3 * channels, channels)
		else:
			self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)  # 1x1卷积

	def forward(self, image, message, mask):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)

		# intermediate1 = torch.cat((intermediate1, mask), dim=1)
		# Mask-D: Grad_cam
		if mask is not None:
			mask = mask.expand_as(intermediate1)
			intermediate1 = torch.cat((intermediate1, mask), dim=1)

		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		return output


class Encoder_MP_Diffusion(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64, diffusion_length=256):
		super(Encoder_MP_Diffusion, self).__init__()
		self.H = H
		self.W = W

		self.diffusion_length = diffusion_length
		self.diffusion_size = int(diffusion_length ** 0.5)

		self.image_pre_layer = ConvBNRelu(3, channels)
		self.image_first_layer = SENet(channels, channels, blocks=blocks)

		self.message_duplicate_layer = nn.Linear(message_length, self.diffusion_length)  # 全连接层
		self.message_pre_layer_0 = ConvBNRelu(1, channels)
		self.message_pre_layer_1 = ExpandNet(channels, channels, blocks=3)
		self.message_pre_layer_2 = SENet(channels, channels, blocks=1)
		self.message_first_layer = SENet(channels, channels, blocks=blocks)

		self.after_concat_layer = ConvBNRelu(2 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)

	def forward(self, image, message):
		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor (with diffusion)
		message_duplicate = self.message_duplicate_layer(message)
		message_image = message_duplicate.view(-1, 1, self.diffusion_size, self.diffusion_size)
		message_pre_0 = self.message_pre_layer_0(message_image)
		message_pre_1 = self.message_pre_layer_1(message_pre_0)
		message_pre_2 = self.message_pre_layer_2(message_pre_1)
		intermediate2 = self.message_first_layer(message_pre_2)

		# concatenate
		concat1 = torch.cat([intermediate1, intermediate2], dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat([intermediate3, image], dim=1)

		# last Conv part of Network
		output = self.final_layer(concat2)

		return output


class Encoder_Mask(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Mask, self).__init__()
		self.H = H
		self.W = W

		# 128*128 64
		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))  # log(16)=4
		message_se_blocks = max(blocks - message_convT_blocks, 1)  # 1

		self.image_pre_layer = ConvBNRelu(3, channels)  # 归一化 3*H*W -> C*H*W
		self.image_first_layer = SENet(channels, channels, blocks=blocks)  #

		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),  # 依据channel number扩展
			SENet(channels, channels, blocks=message_se_blocks),  # SE特征提取
		)

		self.message_first_layer = SENet(channels, channels, blocks=blocks)  # image feature学习

		# self.after_concat_layer = ConvBNRelu(2 * channels, channels)
		# Modify: Mask 3*channel
		self.after_concat_layer = ConvBNRelu(3 * channels, channels)

		self.final_layer = nn.Conv2d(channels + 3, 3, kernel_size=1)  # 1x1卷积

		# Mask-B: Add NLMask [attention mask]
		# self.g_mask = nn.Sequential(
		# 	Non_local_Block(channels, channels // 2),
		# 	ResBlock(channels, channels, 3, 1, 1),
		# 	ResBlock(channels, channels, 3, 1, 1),
		# 	ResBlock(channels, channels, 3, 1, 1),
		# 	nn.Conv2d(channels, channels, 1, 1, 0)
		# )

		# Mask-A: Add NLMask [INN mask]
		self.conv1 = nn.Conv2d(3, channels, 3, 1, 1)  # add channels
		self.conv2 = nn.Conv2d(3 + channels, channels, 3, 1, 1)
		self.conv3 = nn.Conv2d(3 + 2 * channels, channels, 3, 1, 1)
		self.lrelu = nn.LeakyReLU(inplace=True)
		self.senet = SENet(3 + 2 * channels, 3 + 2 * channels, blocks=blocks)

	def forward(self, image, message):
		# Mask-A: generate mask
		x1 = self.lrelu(self.conv1(image))
		x2 = self.lrelu(self.conv2(torch.cat((image, x1), 1)))
		x = torch.cat((image, x1, x2), 1)
		x = self.senet(x)
		mask = self.conv3(x)
		# # print(type(mask))
		# # add non_local
		# mask = self.nl(mask)

		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(intermediate1.size()) (64, H, W)

		# Mask-B: NL-Mask Processor Concat1
		# mask = torch.sigmoid(self.g_mask(intermediate1))
		# print(mask.size()) (64, H, W)
		# print(type(mask))

		# Mask-C: random mask
		# mask = torch.rand_like(inter mediate1)

		intermediate1 = torch.cat((intermediate1, mask), dim=1)
		# # intermediate1 = intermediate1 + mask

		# concatenate
		concat1 = torch.cat((intermediate1, intermediate2), dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat((intermediate3, image), dim=1)

		# last Conv part of Encoder
		output = self.final_layer(concat2)

		mask = torch.sum(mask, 1, keepdim=True)  # 对各个channel求和得到mask
		return output, mask


class Encoder_Mask_Loss(nn.Module):
	'''
	Insert a watermark into an image
	'''

	def __init__(self, H, W, message_length, blocks=4, channels=64):
		super(Encoder_Mask_Loss, self).__init__()
		self.H = H
		self.W = W

		# 128*128 64
		message_convT_blocks = int(np.log2(H // int(np.sqrt(message_length))))  # log(16)=4
		message_se_blocks = max(blocks - message_convT_blocks, 1)  # 1

		self.image_pre_layer = ConvBNRelu(3, channels)  # 归一化 3*H*W -> C*H*W
		self.image_first_layer = SENet(channels, channels, blocks=blocks)  #

		self.message_pre_layer = nn.Sequential(
			ConvBNRelu(1, channels),
			ExpandNet(channels, channels, blocks=message_convT_blocks),  # 依据channel number扩展
			SENet(channels, channels, blocks=message_se_blocks),  # SE特征提取
		)

		self.message_first_layer = SENet(channels, channels, blocks=blocks)  # image feature学习

		# self.after_concat_layer = ConvBNRelu(2 * channels, channels)
		# Modify: Mask 3*channel
		self.after_concat_layer = ConvBNRelu(3 * channels, channels)

		# Modify:
		self.before_final_layer = nn.Conv2d(channels + 3, channels, kernel_size=1)
		self.final_layer = nn.Conv2d(channels, 3, kernel_size=1)

		# Modify-A: Add NLMask [INN mask]
		# self.conv1 = nn.Conv2d(3, channels, 3, 1, 1)  # add channels
		# self.conv2 = nn.Conv2d(3 + channels, channels, 3, 1, 1)
		# self.conv3 = nn.Conv2d(3 + 2 * channels, channels, 3, 1, 1)
		# self.lrelu = nn.LeakyReLU(inplace=True)
		# self.senet = SENet(3 + 2 * channels, 3 + 2 * channels, blocks=blocks)
		# self.mask_generation = Generate_Mask(H, W, blocks)

	def forward(self, image, message):
		# Modify-A: generate mask
		# x1 = self.lrelu(self.conv1(image))
		# x2 = self.lrelu(self.conv2(torch.cat((image, x1), 1)))
		# x = torch.cat((image, x1, x2), 1)
		# x = self.senet(x)
		# mask = self.conv3(x)

		# # print(type(mask))
		# # add non_local
		# mask = self.nl(mask)

		mask = Generate_Mask

		# first Conv part of Encoder
		image_pre = self.image_pre_layer(image)
		intermediate1 = self.image_first_layer(image_pre)

		# Message Processor
		size = int(np.sqrt(message.shape[1]))
		message_image = message.view(-1, 1, size, size)
		message_pre = self.message_pre_layer(message_image)
		intermediate2 = self.message_first_layer(message_pre)
		# print(intermediate1.size()) (64, H, W)

		intermediate1_mask = torch.cat((intermediate1, mask), dim=1)
		# # intermediate1 = intermediate1 + mask

		# concatenate
		concat1 = torch.cat((intermediate1_mask, intermediate2), dim=1)

		# second Conv part of Encoder
		intermediate3 = self.after_concat_layer(concat1)

		# skip connection
		concat2 = torch.cat((intermediate3, image), dim=1)

		# Modify:
		intermediate4 = self.before_final_layer(concat2)

		# last Conv part of Encoder
		# output = self.final_layer(concat2)
		output = self.final_layer(intermediate4)

		mask = torch.sum(mask, 1, keepdim=True)  # 对各个channel求和得到mask
		return output, mask, intermediate1, intermediate4

