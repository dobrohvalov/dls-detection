import os
import cv2
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import time
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics YOLO not available. Using mock predictions.")


@dataclass
class DetectionResult:
    """Результат детекции объекта"""
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": float(self.confidence),
            "bbox": [float(self.bbox[0]), float(self.bbox[1]), 
                    float(self.bbox[2]), float(self.bbox[3])]
        }


@dataclass
class PredictionResult:
    """Полный результат предсказания для изображения"""
    image_id: str
    detections: List[DetectionResult]
    processing_time: float
    model_name: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "image_id": self.image_id,
            "detections": [det.to_dict() for det in self.detections],
            "processing_time": self.processing_time,
            "model_name": self.model_name,
            "total_objects": len(self.detections)
        }


class ModelManager:
    """Менеджер моделей для загрузки и переключения между моделями"""
    
    def __init__(self, models_dir: str = "src/ml_module"):
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.current_model_name: Optional[str] = None
        self.current_model: Optional[Any] = None
        
        # Классы для детекции автотранспорта (VisDrone)
        self.class_names = {
            0: "pedestrian",
            1: "people",
            2: "bicycle",
            3: "car",
            4: "van",
            5: "truck",
            6: "tricycle",
            7: "awning-tricycle",
            8: "bus",
            9: "motor"
        }
        
        # Доступные модели - используем дообученные веса из results
        self.available_models = {
            "yolov8n": {
                "name": "YOLOv8 Nano",
                "path": os.path.join(models_dir, "data", "results", "yolov8n", "train", "weights", "best.pt"),
                "description": "Быстрая, но менее точная модель (дообученная)",
                "speed": "fast",
                "accuracy": "medium"
            },
            "yolov8m": {
                "name": "YOLOv8 Medium",
                "path": os.path.join(models_dir, "data", "results", "yolov8m", "train", "weights", "best.pt"),
                "description": "Баланс скорости и точности (дообученная)",
                "speed": "medium",
                "accuracy": "good"
            },
            "yolov8l": {
                "name": "YOLOv8 Large",
                "path": os.path.join(models_dir, "data", "results", "yolov8l", "train", "weights", "best.pt"),
                "description": "Точная, но медленная модель (дообученная)",
                "speed": "slow",
                "accuracy": "high"
            }
        }
    
    def load_model(self, model_name: str) -> bool:
        """Загружает модель по имени"""
        if model_name not in self.available_models:
            logger.error(f"Model {model_name} not available")
            return False
        
        model_info = self.available_models[model_name]
        model_path = model_info["path"]
        logger.info(f"Loading model {model_name} from {model_path}")
        
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            if YOLO_AVAILABLE:
                logger.info(f"Loading model {model_name} from {model_path}")
                model = YOLO(model_path)
                self.models[model_name] = model
                self.current_model_name = model_name
                self.current_model = model
                logger.info(f"Model {model_name} loaded successfully")
                return True
            else:
                logger.warning("YOLO not available, using mock model")
                self.models[model_name] = "mock"
                self.current_model_name = model_name
                self.current_model = "mock"
                return True
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def switch_model(self, model_name: str) -> bool:
        """Переключает на другую модель"""
        if model_name == self.current_model_name:
            return True
        
        if model_name not in self.models:
            return self.load_model(model_name)
        
        self.current_model_name = model_name
        self.current_model = self.models[model_name]
        logger.info(f"Switched to model {model_name}")
        return True
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Возвращает список доступных моделей"""
        models_list = []
        for model_id, info in self.available_models.items():
            models_list.append({
                "id": model_id,
                "name": info["name"],
                "description": info["description"],
                "speed": info["speed"],
                "accuracy": info["accuracy"],
                "loaded": model_id in self.models,
                "is_current": model_id == self.current_model_name
            })
        return models_list
    
    def get_current_model_info(self) -> Optional[Dict[str, Any]]:
        """Возвращает информацию о текущей модели"""
        if not self.current_model_name:
            return None
        
        info = self.available_models.get(self.current_model_name, {})
        return {
            "id": self.current_model_name,
            "name": info.get("name", ""),
            "description": info.get("description", ""),
            "speed": info.get("speed", ""),
            "accuracy": info.get("accuracy", ""),
            "loaded": True
        }


class MLService:
    """Сервис для обработки ML-запросов"""
    
    def __init__(self, models_dir: str = "src/ml_module"):
        self.model_manager = ModelManager(models_dir)
        self.default_model = "yolov8n"
        
        # Загружаем модель по умолчанию
        if not self.model_manager.load_model(self.default_model):
            logger.warning(f"Failed to load default model {self.default_model}")
    
    def process_image(self, image_data: bytes, model_name: Optional[str] = None) -> PredictionResult:
        """Обрабатывает изображение и возвращает результаты детекции"""
        start_time = time.time()
        
        # Выбираем модель
        if model_name and model_name != self.model_manager.current_model_name:
            self.model_manager.switch_model(model_name)
        
        model_name = self.model_manager.current_model_name or self.default_model
        
        try:
            # Конвертируем bytes в изображение
            image = self._bytes_to_image(image_data)
            if image is None:
                raise ValueError("Failed to decode image")
            
            # Генерируем ID изображения
            image_id = self._generate_image_id(image_data)
            
            # Выполняем предсказание
            if YOLO_AVAILABLE and self.model_manager.current_model != "mock":
                detections = self._predict_with_yolo(image)
            else:
                detections = self._mock_predict(image)
            
            processing_time = time.time() - start_time
            
            result = PredictionResult(
                image_id=image_id,
                detections=detections,
                processing_time=processing_time,
                model_name=model_name
            )
            
            logger.info(f"Processed image {image_id} with {len(detections)} detections "
                       f"in {processing_time:.2f}s using {model_name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            raise
    
    def _bytes_to_image(self, image_data: bytes) -> Optional[np.ndarray]:
        """Конвертирует bytes в изображение OpenCV"""
        try:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return None
    
    def _generate_image_id(self, image_data: bytes) -> str:
        """Генерирует уникальный ID для изображения"""
        import hashlib
        return hashlib.md5(image_data).hexdigest()[:16]
    
    def _predict_with_yolo(self, image: np.ndarray) -> List[DetectionResult]:
        """Выполняет предсказание с использованием YOLO"""
        if self.model_manager.current_model is None:
            return []
        
        try:
            # Выполняем предсказание
            results = self.model_manager.current_model(image, verbose=False)
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.cpu().numpy()
                    for i in range(len(boxes)):
                        box = boxes[i]
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        bbox = box.xyxy[0].tolist()  # x1, y1, x2, y2
                        
                        # Фильтруем только классы автотранспорта (3-9)
                        if 3 <= class_id <= 9:
                            class_name = self.model_manager.class_names.get(class_id, f"class_{class_id}")
                            detection = DetectionResult(
                                class_id=class_id,
                                class_name=class_name,
                                confidence=confidence,
                                bbox=tuple(bbox)
                            )
                            detections.append(detection)
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO prediction error: {e}")
            return []
    
    def _mock_predict(self, image: np.ndarray) -> List[DetectionResult]:
        """Моковое предсказание для тестирования"""
        # Генерируем случайные детекции для тестирования
        import random
        
        height, width = image.shape[:2]
        detections = []
        
        # Классы автотранспорта
        vehicle_classes = [
            (3, "car"),
            (4, "van"),
            (5, "truck"),
            (6, "tricycle"),
            (7, "awning-tricycle"),
            (8, "bus"),
            (9, "motor")
        ]
        
        # Генерируем 3-8 случайных детекций
        num_detections = random.randint(3, 8)
        for _ in range(num_detections):
            class_id, class_name = random.choice(vehicle_classes)
            confidence = random.uniform(0.5, 0.95)
            
            # Случайный bounding box
            box_width = random.randint(50, 200)
            box_height = random.randint(50, 200)
            x1 = random.randint(0, width - box_width)
            y1 = random.randint(0, height - box_height)
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            detection = DetectionResult(
                class_id=class_id,
                class_name=class_name,
                confidence=confidence,
                bbox=(float(x1), float(y1), float(x2), float(y2))
            )
            detections.append(detection)
        
        return detections
    
    def draw_detections(self, image_data: bytes, detections: List[DetectionResult]) -> bytes:
        """Рисует bounding boxes на изображении и возвращает bytes"""
        try:
            image = self._bytes_to_image(image_data)
            if image is None:
                return image_data
            
            # Конвертируем обратно в BGR для OpenCV
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Цвета для разных классов
            colors = {
                3: (0, 255, 0),    # car - зеленый
                4: (255, 0, 0),    # van - синий
                5: (0, 0, 255),    # truck - красный
                6: (255, 255, 0),  # tricycle - голубой
                7: (255, 0, 255),  # awning-tricycle - розовый
                8: (0, 255, 255),  # bus - желтый
                9: (128, 0, 128)   # motor - фиолетовый
            }
            
            for detection in detections:
                color = colors.get(detection.class_id, (255, 255, 255))
                x1, y1, x2, y2 = map(int, detection.bbox)
                
                # Рисуем bounding box
                cv2.rectangle(image_bgr, (x1, y1), (x2, y2), color, 2)
                
                # Рисуем label
                label = f"{detection.class_name}: {detection.confidence:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    image_bgr,
                    (x1, y1 - label_height - baseline - 5),
                    (x1 + label_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    image_bgr,
                    label,
                    (x1, y1 - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1
                )
            
            # Конвертируем обратно в bytes
            _, buffer = cv2.imencode('.jpg', image_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return buffer.tobytes()
            
        except Exception as e:
            logger.error(f"Error drawing detections: {e}")
            return image_data


# Глобальный экземпляр ML-сервиса
ml_service = MLService()