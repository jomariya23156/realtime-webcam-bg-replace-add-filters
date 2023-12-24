import cv2
import torch
import numpy as np
from PIL import Image

from transformers import YolosForObjectDetection, YolosImageProcessor
from transformers import MobileViTFeatureExtractor, MobileViTForSemanticSegmentation
from pydantic_models import Object, Objects

import mediapipe as mp

class ObjectDetection:
    image_processor: YolosImageProcessor | None = None
    model: YolosForObjectDetection | None = None

    def load_model(self) -> None:
        self.image_processor = YolosImageProcessor.from_pretrained("hustvl/yolos-tiny")
        self.model = YolosForObjectDetection.from_pretrained("hustvl/yolos-tiny")

    def predict(self, image:Image.Image) -> Objects:
        if not self.image_processor or not self.model:
            raise RuntimeError("Model is not loaded")
        inputs = self.image_processor(images=image, return_tensors='pt')
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes
        )[0]

        objects: list[Object] = []
        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            if score > 0.6:
                box_values = box.tolist()
                label = self.model.config.id2label[label.item()]
                print('Detected:', label)
                objects.append(Object(box=box_values, label=label))
        return Objects(objects=objects)
    
class ObjectSegmentation:
    image_processor: MobileViTFeatureExtractor | None = None
    model: MobileViTForSemanticSegmentation | None = None
    
    def load_model(self) -> None:
        self.image_processor = MobileViTFeatureExtractor.from_pretrained("apple/deeplabv3-mobilevit-xx-small")
        self.model = MobileViTForSemanticSegmentation.from_pretrained("apple/deeplabv3-mobilevit-xx-small")

    def predict(self, image: np.ndarray) -> np.ndarray:
        if not self.image_processor or not self.model:
            raise RuntimeError("Model is not loaded")
        inputs = self.image_processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)   
        
        target_sizes = [image.shape[:2]]
        results = self.image_processor.post_process_semantic_segmentation(outputs, target_sizes=target_sizes)[0]
        return results
    
class MPSelfieSegmentation:
    model: mp.solutions.selfie_segmentation.SelfieSegmentation | None = None

    def load_model(self) -> None:
        self.model = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

    def predict(self, image: np.ndarray, thr: float=0.5) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model is not loaded")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = self.model.process(image)
        mask = result.segmentation_mask > thr
        return mask