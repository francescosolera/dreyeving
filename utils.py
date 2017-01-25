from keras.layers import Input, Flatten, merge
from keras.models import Model, Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Convolution3D, MaxPooling3D, Convolution2D, UpSampling2D
from keras.layers.core import Reshape
import os
from os.path import join
import sys
import cv2
import numpy as np

# parameters (no need to edit)
t, c, w, h = 16, 3, 112, 112
upsample = 4


def getCoarse2FineModel(summary=True):

    # defined input
    videoclip_cropped = Input((c, t, h, w), name='input1')
    videoclip_original = Input((c, t, h, w), name='input2')
    last_frame_bigger = Input((c, h*upsample, w*upsample), name='input3')

    # coarse saliency model
    coarse_saliency_model = Sequential()
    coarse_saliency_model.add(Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same', name='conv1', subsample=(1, 1, 1), input_shape=(c, t, h, w)))
    coarse_saliency_model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2), border_mode='valid', name='pool1'))
    coarse_saliency_model.add(Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same', name='conv2', subsample=(1, 1, 1)))
    coarse_saliency_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool2'))
    coarse_saliency_model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3a', subsample=(1, 1, 1)))
    coarse_saliency_model.add(Convolution3D(256, 3, 3, 3, activation='relu', border_mode='same', name='conv3b', subsample=(1, 1, 1)))
    coarse_saliency_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2), border_mode='valid', name='pool3'))
    coarse_saliency_model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4a', subsample=(1, 1, 1)))
    coarse_saliency_model.add(Convolution3D(512, 3, 3, 3, activation='relu', border_mode='same', name='conv4b', subsample=(1, 1, 1)))
    coarse_saliency_model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(4, 2, 2), border_mode='valid', name='pool4'))
    coarse_saliency_model.add(Reshape((512, 7, 7)))
    coarse_saliency_model.add(BatchNormalization())
    coarse_saliency_model.add(Convolution2D(256, 3, 3, init='glorot_uniform', border_mode='same'))
    coarse_saliency_model.add(LeakyReLU(alpha=.001))
    coarse_saliency_model.add(UpSampling2D(size=(2, 2)))
    coarse_saliency_model.add(BatchNormalization())
    coarse_saliency_model.add(Convolution2D(128, 3, 3, init='glorot_uniform', border_mode='same'))
    coarse_saliency_model.add(LeakyReLU(alpha=.001))
    coarse_saliency_model.add(UpSampling2D(size=(2, 2)))
    coarse_saliency_model.add(BatchNormalization())
    coarse_saliency_model.add(Convolution2D(64, 3, 3, init='glorot_uniform', border_mode='same'))
    coarse_saliency_model.add(LeakyReLU(alpha=.001))
    coarse_saliency_model.add(UpSampling2D(size=(2, 2)))
    coarse_saliency_model.add(BatchNormalization())
    coarse_saliency_model.add(Convolution2D(32, 3, 3, init='glorot_uniform', border_mode='same'))
    coarse_saliency_model.add(LeakyReLU(alpha=.001))
    coarse_saliency_model.add(UpSampling2D(size=(2, 2)))
    coarse_saliency_model.add(BatchNormalization())
    coarse_saliency_model.add(Convolution2D(16, 3, 3, init='glorot_uniform', border_mode='same'))
    coarse_saliency_model.add(LeakyReLU(alpha=.001))
    coarse_saliency_model.add(BatchNormalization())
    coarse_saliency_model.add(Convolution2D(1, 3, 3, init='glorot_uniform', border_mode='same'))
    coarse_saliency_model.add(LeakyReLU(alpha=.001))

    # loss on cropped image
    coarse_saliency_cropped = coarse_saliency_model(videoclip_cropped)
    cropped_output = Flatten(name='cropped_output')(coarse_saliency_cropped)

    # coarse-to-fine saliency model and loss
    coarse_saliency_original = coarse_saliency_model(videoclip_original)

    x = UpSampling2D((upsample, upsample), name='coarse_saliency_upsampled')(coarse_saliency_original)  # 112 x 4 = 448
    x = merge([x, last_frame_bigger], mode='concat', concat_axis=1)  # merge the last RGB frame

    x = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(x)
    x = Convolution2D(64, 3, 3, border_mode='same', init='he_normal')(x)
    x = LeakyReLU(alpha=.001)(x)
    x = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(x)
    x = LeakyReLU(alpha=.001)(x)
    x = Convolution2D(32, 3, 3, border_mode='same', init='he_normal')(x)
    x = LeakyReLU(alpha=.001)(x)
    x = Convolution2D(16, 3, 3, border_mode='same', init='he_normal')(x)
    x = LeakyReLU(alpha=.001)(x)
    x = Convolution2D(4, 3, 3, border_mode='same', init='he_normal')(x)
    x = LeakyReLU(alpha=.001)(x)

    fine_saliency_model = Convolution2D(1, 3, 3, border_mode='same', activation='relu')(x)

    # loss on full image
    full_fine_output = Flatten(name='full_fine_output')(fine_saliency_model)

    final_model = Model(input=[videoclip_cropped, videoclip_original, last_frame_bigger],
                        output=[cropped_output, full_fine_output])

    if summary:
        print final_model.summary()

    return final_model


def predict_video(model, folder_in, output_path, mean_frame_path):

    # load frames to predict
    frames = []
    frame_list = os.listdir(folder_in)
    mean_frame = cv2.imread(mean_frame_path)
    for frame_name in frame_list:
        frame = cv2.imread(join(folder_in, frame_name))
        frames.append(frame.astype(np.float32) - mean_frame)
    print 'Done loading frames.'

    # start of prediction
    for i in range(t, len(frames)):

        sys.stdout.write('\r{0}: predicting on frame {1:06d}...'.format(folder_in, i))

        # loading videoclip of t frames
        x = np.array(frames[i - t: i])

        x_last_bigger = cv2.resize(x[-1, :, :, :], (h*upsample,w*upsample))
        x_last_bigger = x_last_bigger.transpose(2, 0, 1)
        x_last_bigger = x_last_bigger[None, :]

        x = np.array([cv2.resize(f, (h, w)) for f in x])
        x = x[None, :]
        x = x.transpose(0, 4, 1, 2, 3).astype(np.float32)

        # predict attentional map on last frame of the videoclip
        res = model.predict_on_batch([x, x, x_last_bigger])
        res = res[1]  # keep only fine output
        res = np.clip(res, a_min=0, a_max=255)

        # normalize attentional map between 0 and 1
        res_norm = ((res / res.max()) * 255).astype(np.uint8)
        res_norm = np.reshape(res_norm, (h*upsample,w*upsample))

        cv2.imwrite(join(output_path, '{0:06d}.png'.format(i)), res_norm)