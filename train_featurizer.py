import os

import cv2
import numpy as np
import argparse
from featurizers.VAEFeaturizer import VAEFeaturizer
from featurizers.TDCFeaturizer import TDCFeaturizer
from featurizers.ForwardModelFeaturizer import ForwardModelFeaturizer


def parsing():
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--featurizer_type', help='Choose from [tdc, vae, forward_model]', default='tdc')
    args_parser.add_argument('--featurizer_save_path', help='Path for saving the featurizer', default='default')
    args_parser.add_argument('--initial_width', help='Initial width for the videos', type=int, default=670)
    args_parser.add_argument('--initial_height', help='Initial height for the videos', type=int, default=450)
    args_parser.add_argument('--desired_width', help='Width for the videos after cropping', type=int, default=670)
    args_parser.add_argument('--desired_height', help='Height for the videos after cropping', type=int, default=450)
    args_parser.add_argument('--num_epochs', help='Number of epochs for training', type=int, default=int(2e1))
    args_parser.add_argument('--batch_size', help='Batch size for training', type=int, default=32)
    args_parser.add_argument('--learning_rate', help='Learning rate for training', type=float, default=0.0001)
    return args_parser.parse_args()


def train_featurizer(restore_model=False):
    args = parsing()

    initial_width, initial_height = args.initial_width, args.initial_height
    desired_width, desired_height = args.desired_width, args.desired_height

    # Prepare dataset
    dataset = generate_dataset('dataset', 0.25, initial_width, initial_height)

    if args.featurizer_type == 'vae':
        featurizer_class = VAEFeaturizer
    elif args.featurizer_type == 'tdc':
        featurizer_class = TDCFeaturizer
    elif args.featurizer_type == 'forward_model':
        featurizer_class = ForwardModelFeaturizer
    else:
        raise TypeError

    featurizer = featurizer_class(initial_width, initial_height, desired_width, desired_height,
                                  feature_vector_size=256, learning_rate=args.learning_rate)

    featurizer_save_path = args.featurizer_save_path
    if restore_model:
        featurizer.load(featurizer_save_path)
        print('Load featurizer parameters successfully that have been fully trained!')
        return featurizer, dataset

    featurizer.train(dataset, args.num_epochs, args.batch_size)

    if featurizer_save_path:
        featurizer.save(featurizer_save_path)

    return featurizer, dataset


def generate_dataset(videos_dir, frame_rate, width, height, show=True):
    print('Video: {}'.format(videos_dir), width, height, frame_rate)

    dataset = []
    for video_name in os.listdir(videos_dir):
        if not video_name.endswith('avi'):
            continue

        video_path = os.path.join(videos_dir, video_name)
        print(video_path)

        frames = []
        video = cv2.VideoCapture(video_path)
        while video.isOpened():
            success, frame = video.read()
            if success:
                frame = preprocess_image(frame, (width, height, 3), show=show)
                frames.append(frame)

                frame_index = video.get(cv2.CAP_PROP_POS_FRAMES)
                video_frame_rate = video.get(cv2.CAP_PROP_FPS)
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_frame_rate // frame_rate)
                last_frame_index = video.get(cv2.CAP_PROP_FRAME_COUNT)
                if frame_index >= last_frame_index:
                    # Video is over
                    break
            else:
                break
        if show:
            video.release()
            cv2.waitKey(1) & 0xFF
            cv2.destroyAllWindows()

        frames = np.stack(frames)
        print(frames.shape)
        if len(frames) >= 64:
            dataset.append(frames)

    print(len(dataset), [data.shape for data in dataset])
    return dataset


def preprocess_image(image, size, show=False):
    width, height, channel = size
    image = cv2.resize(image, (width, height))
    if channel == 2:
        color_style = cv2.COLOR_BGR2GRAY
    else:
        color_style = cv2.IMREAD_COLOR
    image = cv2.cvtColor(image, color_style)
    image = np.array(image, dtype=np.uint8)

    if show:
        cv2.imshow('figure', image)
        cv2.waitKey(1)
    return image


if __name__ == '__main__':
    train_featurizer()
