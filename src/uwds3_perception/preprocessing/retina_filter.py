import cv2

class RetinaFilter(object):
    def __init__(self, config_file_path):
        self.retina_filter = None
        self.config_file_path = config_file_path

    def filter(self, frame):
        filtered_frame = frame.copy()
        if self.retina_filter is None:
            self.retina_filter = cv2.bioinspired.Retina_create((frame.shape[1], frame.shape[0]))
            self.retina_filter.setup(self.config_file_path)
            assert self.retina_filter is not None
        self.retina_filter.applyFastToneMapping(frame, filtered_frame)
        return filtered_frame
