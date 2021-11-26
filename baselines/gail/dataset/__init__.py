"""
Data structure of the input .npz:
the data is save in python dictionary format with keys: 'acs', 'ep_rets', 'rews', 'obs'
the values of each item is a list storing the expert trajectory sequentially
a transition can be: (data['obs'][t], data['acs'][t], data['obs'][t+1]) and get reward data['rews'][t]
"""

import numpy as np
import cv2


class Dset(object):
    def __init__(self, inputs, randomize):
        self.inputs = inputs
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.pointer = None

        self.__init_pointer()

    def __init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs

        if self.pointer + batch_size >= self.num_pairs:
            self.__init_pointer()

        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        self.pointer = end
        return inputs


def generate_dataset(videos_path, picture_size, frame_rate=1.0, show_image=False):
    """Converts videos from specified path to ndarrays of shape [numberOfVideos, -1, width, height, 1]

    Args:
        videos_path: Inside the 'videos/' directory, the name of the subdirectory for videos.
        frame_rate: The desired frame rate of the dataset.
        picture_size: Width, height, channel
        show_image:
    Returns:
        The dataset with the new size and framerate, and converted to monochromatic.

    """
    frames = []
    video = cv2.VideoCapture(videos_path)
    while video.isOpened():
        success, frame = video.read()

        if success:
            frame = preprocess_image(frame, picture_size, show=show_image)
            frames.append(frame)

            frame_index = video.get(cv2.CAP_PROP_POS_FRAMES)
            video_frame_rate = video.get(cv2.CAP_PROP_FPS)
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_index + video_frame_rate // frame_rate)
            last_frame_index = video.get(cv2.CAP_PROP_FRAME_COUNT)

            if frame_index >= last_frame_index:  # Video is over
                break
        else:
            break

    frames = np.stack(frames)
    print('data_set shape:', frames.shape)
    return Dset(frames, randomize=True)


def preprocess_image(image, size, show=False):
    """ Changes size, makes image monochromatic """
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


# generate_dataset('C:\\Users\\lydia\\Desktop\\Workspace\\cdr_gail_tracks\\dataset\\gail_video.avi', (670, 450, 3))
