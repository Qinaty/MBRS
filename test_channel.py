from utils import *
from network.Network import *
# from network.Encoder_MP import *
from utils.load_test_channel_setting import *
import numpy as np
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
attention = False
if mask_type == "attention":
    attention = True
    # encoder1 = Encoder_MP.modules()
network = Network(H, W, message_length, noise_layers, device, batch_size, lr, attention, with_diffusion)

EC_path = result_folder + "models/EC_" + str(model_epoch) + ".pth"
network.load_model_ed(EC_path)
dataloader = Dataloader(batch_size, dataset_path, H=H, W=W)
test_dataloader = dataloader.load_test_data()

saved_iterations = [0, 1, 2, 3]
# saved_channels = [1, 18, 25, 27, 38, 43]  # inn
saved_channels = [1, 3, 7, 24, 26, 31, 54]  # concat
saved_channels_none = [1, 6, 18, 21, 28, 55]
saved_all = None

criterion_MSE = nn.MSELoss().to(device)
frequency = np.zeros(64, dtype=np.uint32)

for i, images in enumerate(test_dataloader):
    network.encoder_decoder.eval()
    network.discriminator.eval()

    image = images.to(device)
    message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], message_length))).to(device)
    with torch.no_grad():
        images, messages = images.to(network.device), message.to(network.device)

        if attention:
            encoded_images, mask = network.encoder_decoder.module.encoder(images, messages)
            # encoder_loss = criterion_MSE(images, encoded_images)

            encoded_images1 = images + (encoded_images - image) * strength_factor
            noised_images = network.encoder_decoder.module.noise([encoded_images1, images])

            decoded_messages = network.encoder_decoder.module.decoder(noised_images)
            psnr = kornia.losses.psnr_loss(encoded_images1.detach(), images, 2).item()
            ssim = 1 - 2 * kornia.losses.ssim(encoded_images1.detach(), images, window_size=5, reduction="mean").item()
            error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)
            result = {
                "error_rate": error_rate,
                "psnr": psnr,
                "ssim": ssim,
            }
            print(result)

            # print(mask.shape)
            original_features = network.encoder_decoder.module.encoder.image_pre_layer(image)
            original_features = network.encoder_decoder.module.encoder.image_first_layer(original_features)
            encoded_features = network.encoder_decoder.module.encoder.image_pre_layer(encoded_images)
            encoded_features = network.encoder_decoder.module.encoder.image_first_layer(encoded_features)
            residual_features = (encoded_features - original_features + 1) / 2
            residual_features = residual_features.squeeze(0)

            # mask diff
            # mask = mask.squeeze(0)

            # save encoded_image
            encoded_images2 = encoded_images.cpu()[0]
            # print(encoded_images2.shape)
            filename = os.path.join(save_folder, 'epoch-{}-en.png'.format(i))
            encoded_images2 = (encoded_images2 + 1) / 2
            encoded_images2 = encoded_images2.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            save_en = Image.fromarray(np.array(encoded_images2, dtype=np.uint8)).convert("RGB")
            save_en.save(filename)

            # mask = (mask - mask.min()) / (mask.max() - mask.min())
            # residual_features = (residual_features - residual_features.min()) / (residual_features.max() - residual_features.min())

            # top similar channels
            # print(residual_features.shape)
            channel_num = residual_features.shape[0]
            diff = torch.zeros(channel_num)

            # mask diff
            # for channel_id in range(channel_num):
            #     diff[channel_id] = criterion_MSE(mask[channel_id], residual_features[channel_id])

            # image diff
            for channel_id in range(channel_num):
                diff[channel_id] = torch.mean(residual_features[channel_id])
            topk_channels = torch.topk(diff, 5, largest=False, sorted=True)
            # print(topk_channels)
            topk_index = topk_channels.indices
            topk_index = np.array(topk_index, dtype=np.uint32)

            # print("image{}: ".format(i))
            print(topk_index)

            # for channel_id in range(channel_num):
            #     if channel_id in topk_channels.indices:
            #         frequency[channel_id] += 1

            # print(residual_features.shape)  # (B C W H) [1 64 128 128]
            # save_singel_grey(mask, model_epoch, "mask", i, save_folder)
            # for channel_id in range(channel_num):
            #     if channel_id in saved_channels:
            #         channel_pic = residual_features[channel_id, :, :]
            #         save_singel_grey(channel_pic, model_epoch, channel_id, i, save_folder, "diff")
            #         save_singel_grey(original_features[0, channel_id], model_epoch, channel_id, i, save_folder, "ori")
            #         save_singel_grey(encoded_features[0, channel_id], model_epoch, channel_id, i, save_folder, "en")
        else:
            encoded_images = network.encoder_decoder.module.encoder(images, messages)

            encoded_images1 = images + (encoded_images - image) * strength_factor
            noised_images = network.encoder_decoder.module.noise([encoded_images1, images])
            decoded_messages = network.encoder_decoder.module.decoder(noised_images)
            psnr = kornia.losses.psnr_loss(encoded_images1.detach(), images, 2).item()
            ssim = 1 - 2 * kornia.losses.ssim(encoded_images1.detach(), images, window_size=5, reduction="mean").item()
            error_rate = network.decoded_message_error_rate_batch(messages, decoded_messages)
            result = {
                "error_rate": error_rate,
                "psnr": psnr,
                "ssim": ssim,
            }
            print(result)

            original_features = network.encoder_decoder.module.encoder.image_pre_layer(image)
            original_features = network.encoder_decoder.module.encoder.image_first_layer(original_features)
            encoded_features = network.encoder_decoder.module.encoder.image_pre_layer(encoded_images)
            encoded_features = network.encoder_decoder.module.encoder.image_first_layer(encoded_features)
            residual_features = (encoded_features - original_features + 1) / 2
            residual_features = residual_features.squeeze(0)
            # print(residual_features.shape)

            # save encoded image
            encoded_images2 = encoded_images.cpu()[0]
            filename = os.path.join(save_folder, 'epoch-{}-en.png'.format(i))
            encoded_images2 = (encoded_images2 + 1) / 2
            encoded_images2 = encoded_images2.permute(1, 2, 0).mul(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
            save_en = Image.fromarray(np.array(encoded_images2, dtype=np.uint8)).convert("RGB")
            save_en.save(filename)

            channel_num = residual_features.shape[0]
            diff = torch.zeros(channel_num)

            for channel_id in range(channel_num):
                diff[channel_id] = torch.mean(residual_features[channel_id])
            topk_channels = torch.topk(diff, 5, largest=False, sorted=True)
            topk_index = topk_channels.indices
            topk_index = np.array(topk_index, dtype=np.uint32)
            # print(topk_channels)

            # save residual images
            # for channel_id in range(channel_num):
            #     if channel_id in saved_channels_none:
            #         channel_pic = residual_features[channel_id, :, :]
            #         save_singel_grey(channel_pic, model_epoch, channel_id, i, save_folder, "diff")
            #         save_singel_grey(original_features[0, channel_id], model_epoch, channel_id, i, save_folder, "ori")
            #         save_singel_grey(encoded_features[0, channel_id], model_epoch, channel_id, i, save_folder, "en")

            # save results
            # print("image{}: ".format(i))
            print(topk_index)
            #
            # for channel_id in range(channel_num):
            #     if channel_id in topk_channels.indices:
            #         frequency[channel_id] += 1



'''
test results
'''
# count = 0
# result_index = np.argsort(frequency)
# frequency = np.sort(frequency)
# content = ""
# for channel_id in range(channel_num):
#     index = channel_num - channel_id - 1
#     content += "channel{:3}: {:5}".format(result_index[index], frequency[index])
#     content += "\n"
#     count += frequency[index]
# print(count)
#
# with open(test_log, "a") as file:
#     file.write(content)
