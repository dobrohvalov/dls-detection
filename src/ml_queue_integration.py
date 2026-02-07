"""
Интеграция ML-сервиса с системой очередей
"""
import threading
import time
from typing import Dict, Any

import loguru
from loguru import logger

from src.queue_manager import queue_manager, QueueType
from src.ml_service import ml_service


class MLQueueIntegration:
    """Интеграция ML-сервиса с очередями"""
    
    def __init__(self):
        self.queue_name = "ml_predictions"
        self.is_initialized = False
        
    def initialize(self):
        """Инициализирует очередь и запускает рабочих"""
        if self.is_initialized:
            return
        
        # Создаем очередь для ML-предсказаний (FIFO по умолчанию)
        queue_manager.create_queue(
            name=self.queue_name,
            queue_type=QueueType.FIFO,
            max_size=50
        )
        
        # Запускаем рабочих для обработки очереди
        queue_manager.start_workers(
            queue_name=self.queue_name,
            process_func=self._process_prediction_task,
            num_workers=2
        )
        
        self.is_initialized = True
        logger.info(f"ML queue integration initialized with queue '{self.queue_name}'")
    
    def submit_prediction_task(self, image_data: bytes, model_name: str = None) -> str:
        """
        Отправляет задачу на предсказание в очередь
        
        Args:
            image_data: bytes изображения
            model_name: имя модели (опционально)
            
        Returns:
            ID задачи
        """
        if not self.is_initialized:
            self.initialize()
        
        # Подготавливаем данные задачи
        task_data = {
            "image_data": image_data.hex(),  # Конвертируем bytes в hex для сериализации
            "model_name": model_name,
            "timestamp": time.time()
        }
        
        # Добавляем задачу в очередь
        task_queue = queue_manager.get_queue(self.queue_name)
        if not task_queue:
            raise ValueError(f"Queue '{self.queue_name}' not found")
        
        task_id = task_queue.add_task(task_data)
        logger.info(f"Prediction task {task_id} submitted to queue")
        
        return task_id
    
    def get_task_status(self, task_id: str):
        """Получает статус задачи"""
        task_queue = queue_manager.get_queue(self.queue_name)
        if not task_queue:
            return {"error": f"Queue '{self.queue_name}' not found"}
        
        status = task_queue.get_task_status(task_id)
        if not status:
            return {"error": f"Task {task_id} not found"}
        
        return status
    
    def _process_prediction_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает задачу предсказания
        
        Args:
            task_data: данные задачи
            
        Returns:
            Результат предсказания
        """
        try:
            # Восстанавливаем image_data из hex
            image_data = bytes.fromhex(task_data["image_data"])
            model_name = task_data.get("model_name")
            
            # Выполняем предсказание
            result = ml_service.process_image(image_data, model_name)
            loguru.logger.info(f"Prediction task processed successfully: {result.image_id}")
            # Конвертируем результат в словарь
            result_dict = result.to_dict()
            
            # Добавляем аннотированное изображение (base64)
            annotated_image = ml_service.draw_detections(image_data, result.detections)
            result_dict["annotated_image_base64"] = annotated_image.hex()
            
            logger.info(f"Prediction task processed successfully: {result_dict['image_id']}")
            return result_dict
            
        except Exception as e:
            logger.error(f"Error processing prediction task: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Получает статистику очереди"""
        return queue_manager.get_queue_stats(self.queue_name)
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Очищает старые задачи"""
        task_queue = queue_manager.get_queue(self.queue_name)
        if task_queue:
            task_queue.cleanup_old_tasks(max_age_hours)
    
    def process_image_sync(self, image_data: bytes, model_name: str = None, return_image: bool = False) -> Dict[str, Any]:
        """
        Синхронная обработка изображения (без очереди)
        
        Args:
            image_data: bytes изображения
            model_name: имя модели (опционально)
            return_image: если True, возвращает словарь с ключом 'image_bytes' вместо 'annotated_image_base64'
            
        Returns:
            Результат предсказания
        """
        try:
            result = ml_service.process_image(image_data, model_name)
            logger.info(f"Sync prediction processed successfully: {result.image_id}")
            result_dict = result.to_dict()
            
            # Добавляем аннотированное изображение
            annotated_image = ml_service.draw_detections(image_data, result.detections)
            
            if return_image:
                # Возвращаем bytes изображения
                result_dict["image_bytes"] = annotated_image
            else:
                # Возвращаем hex строку для обратной совместимости
                result_dict["annotated_image_base64"] = annotated_image.hex()
            
            return result_dict
            
        except Exception as e:
            logger.error(f"Error in sync processing: {e}")
            return {
                "error": str(e),
                "success": False
            }


# Глобальный экземпляр интеграции
ml_queue_integration = MLQueueIntegration()


def initialize_ml_queue():
    """Функция для инициализации ML очереди при запуске приложения"""
    ml_queue_integration.initialize()
    return ml_queue_integration