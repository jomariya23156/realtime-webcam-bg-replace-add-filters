import cv2
import torch
import numpy as np
import tensorflow as tf
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
    
class CartoonGAN:
    # ref: https://www.kaggle.com/code/kerneler/cartoongan
    model = None
    model_dict = {
        "dr": "./models/CartoonGAN/dr.tflite",
        "int8": "./models/CartoonGAN/int8.tflite",
        "fp16": "./models/CartoonGAN/fp16.tflite"
    }

    def load_model(self, model_type: str='dr') -> None:
        self.model = tf.lite.Interpreter(model_path=self.model_dict[model_type])
        self.model_type = model_type

    def convert_image(self, image: np.ndarray) -> tf.Tensor:
        img = image.copy()
        img = img.astype(np.float32) / 127.5 - 1
        img = np.expand_dims(img, 0)
        img = tf.convert_to_tensor(img)
        return img

    def preprocess_image(self, image: tf.Tensor, target_dim: int=224) -> tf.Tensor:
        # here we don't care about preserving the aspect ratio
        image = tf.image.resize(image, (target_dim, target_dim))
        return image    

    def predict(self, image: np.ndarray) -> np.ndarray:
        if not self.model:
            raise RuntimeError("Model is not loaded")
        tensor_img = self.convert_image(image)
        if self.model_type == "fp16":
            preprocessed_img = self.preprocess_image(tensor_img, target_dim=224) 
        else:
            preprocessed_img = self.preprocess_image(tensor_img, target_dim=512)
        input_details = self.model.get_input_details()
        self.model.allocate_tensors()
        self.model.set_tensor(input_details[0]['index'], preprocessed_img)
        self.model.invoke()
        raw_prediction = self.model.tensor(self.model.get_output_details()[0]['index'])()
        output = (np.squeeze(raw_prediction)+1.0)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output