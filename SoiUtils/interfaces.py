import abc
from typing import Union, Tuple, List
import torch
import numpy as np

class ClassifierPredictor(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        return hasattr(__subclass, 'predict') and callable(__subclass.predict) or NotImplemented
    
    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> Union[List[str], Tuple[List[str], torch.Tensor]]:
        """predict for classes for images"""
        raise NotImplementedError


class ObjectDetectorPredictor(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        return hasattr(__subclass, 'predict') and callable(__subclass.predict) or NotImplemented
    
    @abc.abstractmethod
    def predict(self, *args, **kwargs) -> Tuple[List[str], np.array]:
        """predict bboxes and classes for images"""
        raise NotImplementedError
    
class Resetable(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        return hasattr(__subclass,'reset') and callable(__subclass.reset) or NotImplemented
    
    @abc.abstractmethod
    def reset(self, *args, **kwargs) -> None:
        """reset the object"""
        raise NotImplementedError

 
class Updatable(metaclass=abc.ABCMeta):
    @classmethod
    def __subclasshook__(cls, __subclass: type) -> bool:
        return hasattr(__subclass,'update') and callable(__subclass.update) or NotImplemented
    

    @abc.abstractmethod 
    def update(self,*args, **kwargs) -> None:
        # update the object according to a new configuration
        raise NotImplementedError
        
    

    
