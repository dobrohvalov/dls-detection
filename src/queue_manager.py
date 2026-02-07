import queue
import threading
import uuid
import time
from enum import Enum
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
import json
from loguru import logger


class QueueType(Enum):
    FIFO = "fifo"
    LIFO = "lifo"
    PRIORITY = "priority"


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    """Задача для обработки в очереди"""
    id: str
    data: Dict[str, Any]
    priority: int = 0
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Конвертирует задачу в словарь"""
        task_dict = asdict(self)
        task_dict['status'] = self.status.value
        task_dict['created_at'] = self.created_at.isoformat() if self.created_at else None
        task_dict['started_at'] = self.started_at.isoformat() if self.started_at else None
        task_dict['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return task_dict


class TaskQueue:
    """Управление очередями задач"""
    
    def __init__(self, queue_type: QueueType = QueueType.FIFO, max_size: int = 100):
        self.queue_type = queue_type
        self.max_size = max_size
        
        if queue_type == QueueType.FIFO:
            self._queue = queue.Queue(maxsize=max_size)
        elif queue_type == QueueType.LIFO:
            self._queue = queue.LifoQueue(maxsize=max_size)
        elif queue_type == QueueType.PRIORITY:
            self._queue = queue.PriorityQueue(maxsize=max_size)
        
        self.tasks: Dict[str, Task] = {}
        self.lock = threading.RLock()
        self.workers = []
        self.is_running = False
    
    def add_task(self, data: Dict[str, Any], priority: int = 0) -> str:
        """Добавляет задачу в очередь и возвращает её ID"""
        with self.lock:
            task_id = str(uuid.uuid4())
            task = Task(
                id=task_id,
                data=data,
                priority=priority
            )
            
            if self.queue_type == QueueType.PRIORITY:
                self._queue.put((priority, task_id))
            else:
                self._queue.put(task_id)
            
            self.tasks[task_id] = task
            logger.info(f"Task {task_id} added to queue")
            return task_id
    
    def get_task(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Task]:
        """Получает следующую задачу из очереди"""
        try:
            if self.queue_type == QueueType.PRIORITY:
                item = self._queue.get(block=block, timeout=timeout)
                if item:
                    priority, task_id = item
            else:
                task_id = self._queue.get(block=block, timeout=timeout)
            
            with self.lock:
                task = self.tasks.get(task_id)
                if task:
                    task.status = TaskStatus.PROCESSING
                    task.started_at = datetime.now()
            return task
        except queue.Empty:
            return None
    
    def complete_task(self, task_id: str, result: Dict[str, Any] = None, error: str = None):
        """Отмечает задачу как завершенную"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                task.completed_at = datetime.now()
                if error:
                    task.status = TaskStatus.FAILED
                    task.error = error
                else:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                logger.info(f"Task {task_id} completed with status {task.status.value}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Получает статус задачи"""
        with self.lock:
            task = self.tasks.get(task_id)
            if task:
                return task.to_dict()
        return None
    
    def get_all_tasks(self) -> List[Dict[str, Any]]:
        """Получает все задачи"""
        with self.lock:
            return [task.to_dict() for task in self.tasks.values()]
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Удаляет старые задачи"""
        with self.lock:
            now = datetime.now()
            to_remove = []
            for task_id, task in self.tasks.items():
                if task.completed_at:
                    age_hours = (now - task.completed_at).total_seconds() / 3600
                    if age_hours > max_age_hours:
                        to_remove.append(task_id)
            
            for task_id in to_remove:
                del self.tasks[task_id]
            logger.info(f"Cleaned up {len(to_remove)} old tasks")


class Worker(threading.Thread):
    """Рабочий поток для обработки задач"""
    
    def __init__(self, task_queue: TaskQueue, process_func, worker_id: int):
        super().__init__(daemon=True)
        self.task_queue = task_queue
        self.process_func = process_func
        self.worker_id = worker_id
        self.is_working = False
    
    def run(self):
        """Основной цикл рабочего потока"""
        logger.info(f"Worker {self.worker_id} started")
        while True:
            try:
                task = self.task_queue.get_task(block=True, timeout=1)
                if task:
                    self.is_working = True
                    logger.info(f"Worker {self.worker_id} processing task {task.id}")
                    
                    try:
                        result = self.process_func(task.data)
                        self.task_queue.complete_task(task.id, result=result)
                    except Exception as e:
                        logger.error(f"Worker {self.worker_id} failed to process task {task.id}: {e}")
                        self.task_queue.complete_task(task.id, error=str(e))
                    
                    self.is_working = False
            except Exception as e:
                logger.error(f"Worker {self.worker_id} error: {e}")
                time.sleep(1)


class QueueManager:
    """Менеджер очередей для всего приложения"""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.queues: Dict[str, TaskQueue] = {}
        self.workers: Dict[str, List[Worker]] = {}
        self._initialized = True
    
    def create_queue(self, name: str, queue_type: QueueType = QueueType.FIFO, 
                    max_size: int = 100) -> TaskQueue:
        """Создает новую очередь"""
        if name in self.queues:
            return self.queues[name]
        
        task_queue = TaskQueue(queue_type=queue_type, max_size=max_size)
        self.queues[name] = task_queue
        logger.info(f"Queue '{name}' created with type {queue_type.value}")
        return task_queue
    
    def get_queue(self, name: str) -> Optional[TaskQueue]:
        """Получает очередь по имени"""
        return self.queues.get(name)
    
    def start_workers(self, queue_name: str, process_func, num_workers: int = 2):
        """Запускает рабочих для обработки очереди"""
        task_queue = self.get_queue(queue_name)
        if not task_queue:
            raise ValueError(f"Queue '{queue_name}' not found")
        
        workers = []
        for i in range(num_workers):
            worker = Worker(task_queue, process_func, worker_id=i)
            worker.start()
            workers.append(worker)
        
        self.workers[queue_name] = workers
        logger.info(f"Started {num_workers} workers for queue '{queue_name}'")
    
    def stop_workers(self, queue_name: str):
        """Останавливает рабочих (рабочие потоки демонические, остановятся при завершении программы)"""
        if queue_name in self.workers:
            logger.info(f"Workers for queue '{queue_name}' will stop when program exits")
    
    def get_queue_stats(self, queue_name: str) -> Dict[str, Any]:
        """Получает статистику по очереди"""
        task_queue = self.get_queue(queue_name)
        if not task_queue:
            return {}
        
        with task_queue.lock:
            pending = task_queue._queue.qsize()
            total_tasks = len(task_queue.tasks)
            completed = sum(1 for t in task_queue.tasks.values() 
                          if t.status == TaskStatus.COMPLETED)
            failed = sum(1 for t in task_queue.tasks.values() 
                        if t.status == TaskStatus.FAILED)
            processing = sum(1 for t in task_queue.tasks.values() 
                           if t.status == TaskStatus.PROCESSING)
            
            return {
                "queue_name": queue_name,
                "queue_type": task_queue.queue_type.value,
                "pending_tasks": pending,
                "total_tasks": total_tasks,
                "completed_tasks": completed,
                "failed_tasks": failed,
                "processing_tasks": processing,
                "max_size": task_queue.max_size
            }


# Глобальный экземпляр менеджера очередей
queue_manager = QueueManager()