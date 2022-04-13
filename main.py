# Demo - train the DenseFuse network & use it to generate an image

from __future__ import print_function

from train_recons import train_recons
from generate import generate
from utils import list_images

import time
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# True for training phase
IS_TRAINING = True
# True for video sequences(frames)
IS_VIDEO = False
# True for RGB images
IS_RGB = True

# True - 1000 images for validation
# This is a very time-consuming operation when TRUE
IS_Validation = False

BATCH_SIZE = 2 #2
EPOCHES = 1 #4

SSIM_WEIGHTS = [1, 10, 100, 1000]
MODEL_SAVE_PATHS = [
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e1.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e2.ckpt',
    './models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e3.ckpt',
]

# MODEL_SAVE_PATH = './models/deepfuse_dense_model_bs4_epoch2_relu_pLoss_noconv_test.ckpt'
# model_pre_path  = './models/deepfuse_dense_model_bs2_epoch2_relu_pLoss_noconv_NEW.ckpt'

# In testing process, 'model_pre_path' is set to None
# The "model_pre_path" in "main.py" is just a pre-train model and not necessary for training and testing. 
# It is set as None when you want to train your own model. 
# If you already train a model, you can set it as your model for initialize weights.
model_pre_path = None#'./models/densefuse_gray/densefuse_model_bs2_epoch4_all_weight_1e0.ckpt'

def main():

	if IS_TRAINING:
		original_imgs_path = list_images('./validation/validation/')#D:/Dataset/Image_fusion_MSCOCO/train2014/
		validatioin_imgs_path = list_images('./validation/validation/')
		for ssim_weight, model_save_path in zip(SSIM_WEIGHTS, MODEL_SAVE_PATHS):
			print('\nBegin to train the network ...\n')
			train_recons(original_imgs_path, validatioin_imgs_path, model_save_path, model_pre_path, ssim_weight, EPOCHES, BATCH_SIZE, IS_Validation, debug=True)

			print('\nSuccessfully! Done training...\n')
	else:
		if IS_VIDEO:
			ssim_weight = SSIM_WEIGHTS[0]
			model_path = MODEL_SAVE_PATHS[0]

			IR_path = list_images('video/1_IR/')
			VIS_path = list_images('video/1_VIS/')
			output_save_path = 'video/fused'+ str(ssim_weight) +'/'
			generate(IR_path, VIS_path, model_path, model_pre_path,
			         ssim_weight, 0, IS_VIDEO, 'addition', output_path=output_save_path)
		else:#start
			start_time = time.time()                                     
			ssim_weight = SSIM_WEIGHTS[0]
			model_path = MODEL_SAVE_PATHS[0]
			print('\nBegin to generate pictures ...\n')
			# path = 'images/IV_images/'MR-T1_PET_images
			path = 'images/MR-T1_PET_images/'
			for i in range(9):
				index = i + 1
				infrared = path + 'MR' + str(index) + '.png'
				visible = path + 'PET' + str(index) + '.png'

				# RGB images
				#infrared = path + 'lytro-2-A.jpg'
				#visible = path + 'lytro-2-B.jpg'

				# choose fusion layer
				fusion_type = 'addition'
				# fusion_type = 'l1'
				# for ssim_weight, model_path in zip(SSIM_WEIGHTS, MODEL_SAVE_PATHS):
				# 	output_save_path = 'outputs'
                #
				# 	generate(infrared, visible, model_path, model_pre_path,
				# 	         ssim_weight, index, IS_VIDEO, is_RGB, type = fusion_type, output_path = output_save_path)

				output_save_path = 'outputs'
				generate(infrared, visible, model_path, model_pre_path,
						 ssim_weight, index, IS_VIDEO, IS_RGB, type = fusion_type, output_path = output_save_path)

			end_time = time.time()
			print('totally cost',end_time-start_time)

if __name__ == '__main__':
    main()

