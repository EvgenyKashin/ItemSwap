import os
import keras.backend as K
import tensorflow as tf
import os
import cv2
import glob
import time
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from keras_vggface.vggface import VGGFace
from scripts_faces.networks.faceswap_gan_model import FaceswapGANModel
from scripts_faces.data_loader.data_loader import DataLoader
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--iters', default='40000')
    parser.add_argument('--cuda_device', default='0')
    args = parser.parse_args()

    iters = int(args.iters)
    cuda_device = args.cuda_device
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)

    K.set_session(session)

    data_folder_path = 'data_faces'
    K.set_learning_phase(1)

    num_cpus = os.cpu_count() // 2

    # Input/Output resolution
    RESOLUTION = 128  # 64x64, 128x128, 256x256
    assert (RESOLUTION % 64) == 0, "RESOLUTION should be 64, 128, or 256."

    # Batch size
    batchSize = 8
    assert (batchSize != 1 and batchSize % 2 == 0) , "batchSize should be an even number."

    # Use motion blurs (data augmentation)
    # set True if training data contains images extracted from videos
    use_da_motion_blur = True

    # Use eye-aware training
    # require images generated from prep_binary_masks.ipynb
    use_bm_eyes = True

    # Probability of random color matching (data augmentation)
    prob_random_color_match = 0.5

    da_config = {
        "prob_random_color_match": prob_random_color_match,
        "use_da_motion_blur": use_da_motion_blur,
        "use_bm_eyes": use_bm_eyes
    }

    # Path to training images
    img_dirA = f'{data_folder_path}/facesA/aligned_faces/'
    img_dirB = f'{data_folder_path}/facesB/aligned_faces/'
    img_dirA_bm_eyes = f'{data_folder_path}/binary_masks/faceA_eyes'
    img_dirB_bm_eyes = f'{data_folder_path}/binary_masks/faceB_eyes'

    # Path to saved model weights
    models_dir = "weights_faces/gan_models"

    # Architecture configuration
    arch_config = {'IMAGE_SHAPE': (RESOLUTION, RESOLUTION, 3),
                   'use_self_attn': True,
                   'norm': "instancenorm",
                   'model_capacity': "standard"}

    # Loss function weights configuration
    loss_weights = {'w_D': 0.1, 'w_recon': 1., 'w_edge': 0.1, 'w_eyes': 30.,
                    'w_pl': (0.01, 0.1, 0.3, 0.1)}

    # Init. loss config.
    loss_config = {"gan_training": "mixup_LSGAN", 'use_PL': False,
                   "PL_before_activ": False, 'use_mask_hinge_loss': False,
                   'm_mask': 0., 'lr_factor': 1., 'use_cyclic_loss': False}

    model = FaceswapGANModel(**arch_config)
    model.load_weights(path=models_dir)

    # VGGFace ResNet50
    vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
    model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
    model.build_train_functions(loss_weights=loss_weights, **loss_config)

    # Create ./models directory
    Path(models_dir).mkdir(parents=True, exist_ok=True)

    # Get filenames
    train_A = glob.glob(img_dirA+"/*.*")
    train_B = glob.glob(img_dirB+"/*.*")

    train_AnB = train_A + train_B

    assert len(train_A), "No image found in " + str(img_dirA)
    assert len(train_B), "No image found in " + str(img_dirB)
    print("Number of images in folder A: " + str(len(train_A)))
    print("Number of images in folder B: " + str(len(train_B)))

    if use_bm_eyes:
        assert len(glob.glob(img_dirA_bm_eyes+"/*.*")),\
            "No binary mask found in " + str(img_dirA_bm_eyes)
        assert len(glob.glob(img_dirB_bm_eyes+"/*.*")),\
            "No binary mask found in " + str(img_dirB_bm_eyes)
        assert len(glob.glob(img_dirA_bm_eyes+"/*.*")) == len(train_A),\
            "Number of faceA images does not match number of their binary masks." \
            " Can be caused by any none image file in the folder."
        assert len(glob.glob(img_dirB_bm_eyes+"/*.*")) == len(train_B),\
            "Number of faceB images does not match number of their binary masks." \
            " Can be caused by any none image file in the folder."


    def show_loss_config(loss_config):
        for config, value in loss_config.items():
            print(f"{config} = {value}")


    def reset_session(save_path):
        global model, vggface
        global train_batchA, train_batchB
        model.save_weights(path=save_path)
        del model
        del vggface
        del train_batchA
        del train_batchB
        K.clear_session()
        model = FaceswapGANModel(**arch_config)
        model.load_weights(path=save_path)
        vggface = VGGFace(include_top=False, model='resnet50', input_shape=(224, 224, 3))
        model.build_pl_model(vggface_model=vggface, before_activ=loss_config["PL_before_activ"])
        train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                                  RESOLUTION, num_cpus, K.get_session(), **da_config)
        train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                                  RESOLUTION, num_cpus, K.get_session(), **da_config)


    def get_transpose_axes( n ):
        if n % 2 == 0:
            y_axes = list(range(1, n-1, 2))
            x_axes = list(range(0, n-1, 2))
        else:
            y_axes = list(range(0, n-1, 2))
            x_axes = list(range(1, n-1, 2))
        return y_axes, x_axes, [n-1]


    def stack_images(images):
        images_shape = np.array(images.shape)
        new_axes = get_transpose_axes(len(images_shape))
        new_shape = [np.prod(images_shape[x]) for x in new_axes]
        return np.transpose(
            images,
            axes = np.concatenate(new_axes)
            ).reshape(new_shape)


    def showG(test_A, test_B, path_A, path_B, batchSize):
        figure_A = np.stack([
            test_A,
            np.squeeze(np.array([path_A([test_A[i:i+1]])
                                 for i in range(test_A.shape[0])])),
            np.squeeze(np.array([path_B([test_A[i:i+1]])
                                 for i in range(test_A.shape[0])])),
            ], axis=1)
        figure_B = np.stack([
            test_B,
            np.squeeze(np.array([path_B([test_B[i:i+1]])
                                 for i in range(test_B.shape[0])])),
            np.squeeze(np.array([path_A([test_B[i:i+1]])
                                 for i in range(test_B.shape[0])])),
            ], axis=1)

        figure = np.concatenate([figure_A, figure_B], axis=0)
        figure = figure.reshape((4, batchSize//2) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        return figure


    def showG_mask(test_A, test_B, path_A, path_B, batchSize):
        figure_A = np.stack([
            test_A,
            (np.squeeze(np.array([path_A([test_A[i:i+1]])
                                  for i in range(test_A.shape[0])])))*2-1,
            (np.squeeze(np.array([path_B([test_A[i:i+1]])
                                  for i in range(test_A.shape[0])])))*2-1,
            ], axis=1)
        figure_B = np.stack([
            test_B,
            (np.squeeze(np.array([path_B([test_B[i:i+1]])
                                  for i in range(test_B.shape[0])])))*2-1,
            (np.squeeze(np.array([path_A([test_B[i:i+1]])
                                  for i in range(test_B.shape[0])])))*2-1,
            ], axis=1)

        figure = np.concatenate([figure_A, figure_B], axis=0)
        figure = figure.reshape((4,batchSize//2) + figure.shape[1:])
        figure = stack_images(figure)
        figure = np.clip((figure + 1) * 255 / 2, 0, 255).astype('uint8')
        figure = cv2.cvtColor(figure, cv2.COLOR_BGR2RGB)
        return figure

    t0 = time.time()
    gen_iterations = 0

    errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
    errGAs = {}
    errGBs = {}
    # Dictionaries are ordered in Python 3.6
    for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
        errGAs[k] = 0
        errGBs[k] = 0

    display_iters = 300
    backup_iters = 3000
    TOTAL_ITERS = 40000

    train_batchA = DataLoader(train_A, train_AnB, batchSize, img_dirA_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)
    train_batchB = DataLoader(train_B, train_AnB, batchSize, img_dirB_bm_eyes,
                              RESOLUTION, num_cpus, K.get_session(), **da_config)

    while gen_iterations <= TOTAL_ITERS:
        # Loss function automation
        if gen_iterations == (TOTAL_ITERS//5 - display_iters//2):
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.0
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (TOTAL_ITERS//5 + TOTAL_ITERS//10 - display_iters//2):
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.5
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Complete.")
        elif gen_iterations == (2*TOTAL_ITERS//5 - display_iters//2):
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.2
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (TOTAL_ITERS//2 - display_iters//2):
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.4
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (2*TOTAL_ITERS//3 - display_iters//2):
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.
            loss_config['lr_factor'] = 0.3
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (8*TOTAL_ITERS//10 - display_iters//2):
            # swap decoders
            model.decoder_A.load_weights("weights_faces/gan_models/decoder_B.h5")
            # swap decoders
            model.decoder_B.load_weights("weights_faces/gan_models/decoder_A.h5")
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = True
            loss_config['m_mask'] = 0.1
            loss_config['lr_factor'] = 0.3
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")
        elif gen_iterations == (9*TOTAL_ITERS//10 - display_iters//2):
            loss_config['use_PL'] = True
            loss_config['use_mask_hinge_loss'] = False
            loss_config['m_mask'] = 0.0
            loss_config['lr_factor'] = 0.1
            reset_session(models_dir)
            print("Building new loss funcitons...")
            show_loss_config(loss_config)
            model.build_train_functions(loss_weights=loss_weights, **loss_config)
            print("Done.")

        if gen_iterations == 5:
            print("working.")

        # Train dicriminators for one batch
        data_A = train_batchA.get_next_batch()
        data_B = train_batchB.get_next_batch()
        errDA, errDB = model.train_one_batch_D(data_A=data_A, data_B=data_B)
        errDA_sum +=errDA[0]
        errDB_sum +=errDB[0]

        # Train generators for one batch
        data_A = train_batchA.get_next_batch()
        data_B = train_batchB.get_next_batch()
        errGA, errGB = model.train_one_batch_G(data_A=data_A, data_B=data_B)
        errGA_sum += errGA[0]
        errGB_sum += errGB[0]
        for i, k in enumerate(['ttl', 'adv', 'recon', 'edge', 'pl']):
            errGAs[k] += errGA[i]
            errGBs[k] += errGB[i]
        gen_iterations+=1

        # Visualization
        if gen_iterations % display_iters == 0:
            # Display loss information
            show_loss_config(loss_config)
            print("----------")
            print('[iter %d] Loss_DA: %f Loss_DB: %f Loss_GA: %f Loss_GB: %f time: %f'
            % (gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
               errGA_sum/display_iters, errGB_sum/display_iters, time.time()-t0))
            print("----------")
            print("Generator loss details:")
            print(f'[Adversarial loss]')
            print(f'GA: {errGAs["adv"]/display_iters:.4f} '
                  f'GB: {errGBs["adv"]/display_iters:.4f}')
            print(f'[Reconstruction loss]')
            print(f'GA: {errGAs["recon"]/display_iters:.4f} '
                  f'GB: {errGBs["recon"]/display_iters:.4f}')
            print(f'[Edge loss]')
            print(f'GA: {errGAs["edge"]/display_iters:.4f} '
                  f'GB: {errGBs["edge"]/display_iters:.4f}')
            if loss_config['use_PL'] == True:
                print(f'[Perceptual loss]')
                try:
                    print(f'GA: {errGAs["pl"][0]/display_iters:.4f} '
                          f'GB: {errGBs["pl"][0]/display_iters:.4f}')
                except:
                    print(f'GA: {errGAs["pl"]/display_iters:.4f} '
                          f'GB: {errGBs["pl"]/display_iters:.4f}')

            # Display images
            print("----------")
            wA, tA, _ = train_batchA.get_next_batch()
            wB, tB, _ = train_batchB.get_next_batch()
            try:
                print("Transformed (masked) results:")
                tr_m = showG(tA, tB, model.path_A, model.path_B, batchSize)
                print("Masks:")
                ma = showG_mask(tA, tB, model.path_mask_A, model.path_mask_B, batchSize)
                print("Reconstruction results:")
                rec = showG(wA, wB, model.path_bgr_A, model.path_bgr_B, batchSize)
                plt.imsave(f'cache/ep_{gen_iterations}.jpg', np.vstack((tr_m, ma, rec)),
                           format="jpg")
            except:
                pass
            errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
            for k in ['ttl', 'adv', 'recon', 'edge', 'pl']:
                errGAs[k] = 0
                errGBs[k] = 0

            # Save models
            model.save_weights(path=models_dir)

        # Backup models
        if gen_iterations % backup_iters == 0:
            bkup_dir = f"{models_dir}/backup_iter{gen_iterations}"
            Path(bkup_dir).mkdir(parents=True, exist_ok=True)
            model.save_weights(path=bkup_dir)
