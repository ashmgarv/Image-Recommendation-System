from abc import ABC, abstractmethod

class Feature(ABC):

    @abstractmethod
    def process_img(self, img_path):
        pass

    @abstractmethod
    def visualize(self, img_path, op_path):
        pass
