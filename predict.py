from utils import getCoarse2FineModel, predict_video
from keras.optimizers import Adam

if __name__ == '__main__':

    output_dir_root = 'out'
    weights_file = 'weights/model_weights.h5'
    dreyeve_data_dir = 'data_sample/54'

    # load model for prediction
    model = getCoarse2FineModel(summary=True)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    model.compile(optimizer=opt,
                  loss={'cropped_output': 'mse', 'full_fine_output': 'mse'},
                  loss_weights={'cropped_output': 1.0, 'full_fine_output': 1.0})

    # load pre-trained weights
    model.load_weights(weights_file)

    # predict on sample data (first 200 frames of run 54 from DR(eye)VE
    predict_video(model, dreyeve_data_dir,
                  output_path=output_dir_root,
                  mean_frame_path='data_sample\dreyeve_mean_frame.png')
