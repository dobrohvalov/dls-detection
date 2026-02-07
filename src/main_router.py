from typing import Annotated, Optional
from fastapi import APIRouter, Depends, Request, Header, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, Response
from loguru import logger
from pydantic import BaseModel
from urllib.parse import parse_qs
from starlette.responses import RedirectResponse, HTMLResponse
import base64
import json

from config import settings
from src.ml_queue_integration import ml_queue_integration, initialize_ml_queue
from src.ml_service import ml_service
from src.queue_manager import queue_manager, QueueType


class CommonQueryParams:
    def __init__(self, skip: int = 0, limit: int = 100):
        self.skip = skip
        self.limit = limit


router_params = {
    "tags": ["Main router, for redirect to front"],
    "responses": {404: {"description": "Not found"}}
}


main_router = APIRouter(**router_params)


# Инициализация ML очереди при импорте модуля
# @main_router.on_event("startup")
# async def startup_event():
#     """Инициализация ML очереди при запуске приложения"""
#     try:
#         initialize_ml_queue()
#         logger.info("ML queue initialized on startup")
#     except Exception as e:
#         logger.error(f"Failed to initialize ML queue: {e}")


# Pydantic модели для запросов
class SwitchModelRequest(BaseModel):
    model_name: str


class PredictionRequest(BaseModel):
    image_base64: Optional[str] = None
    model_name: Optional[str] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    created_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[dict] = None
    error: Optional[str] = None


# Существующие endpoints для Bitrix24
@main_router.head("/", response_description="Check install app for b24")
async def install_post(strange_header: Annotated[str | None, Header(convert_underscores=False)] = None):
    """ Проверочная функция, для проверки возможности установки для б24 """
    logger.debug(strange_header)
    return {"Test": strange_header}



class RequestBodyModel(BaseModel):
    AUTH_ID: str
    AUTH_EXPIRES: int
    REFRESH_ID: str
    member_id: str
    status: str
    PLACEMENT: str
    PLACEMENT_OPTIONS: str



# Новые ML endpoints
@main_router.post("/predict", tags=["ML Prediction"])
async def predict_image(
    file: UploadFile = File(...),
    model_name: Optional[str] = Form(None),
    async_mode: bool = Form(False),
    return_image: bool = Form(False)
):
    """
    Загрузка изображения и получение предсказания
    
    Args:
        file: изображение для обработки
        model_name: имя модели (yolov8n, yolov8m, yolov8l)
        async_mode: использовать асинхронную обработку через очередь
        return_image: если True и async_mode=False, возвращает изображение с bounding boxes
    
    Returns:
        Результат предсказания, ID задачи или изображение
    """
    try:
        # Читаем содержимое файла
        contents = await file.read()
        
        if async_mode:
            # Асинхронный режим: добавляем задачу в очередь
            task_id = ml_queue_integration.submit_prediction_task(contents, model_name)
            return JSONResponse(
                status_code=202,
                content={
                    "message": "Task submitted for processing",
                    "task_id": task_id,
                    "status_endpoint": f"/api/tasks/{task_id}"
                }
            )
        else:
            # Синхронный режим: обрабатываем сразу
            result = ml_queue_integration.process_image_sync(contents, model_name, return_image=return_image)
            
            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])
            
            if return_image and "image_bytes" in result:
                # Возвращаем изображение
                image_bytes = result.pop("image_bytes")  # Удаляем bytes из результата
                return Response(
                    content=image_bytes,
                    media_type="image/jpeg",
                    headers={
                        "X-Detection-Result": json.dumps(result),
                        "X-Image-ID": result.get("image_id", ""),
                        "X-Total-Objects": str(result.get("total_objects", 0))
                    }
                )
            else:
                # Возвращаем JSON с результатом
                return JSONResponse(content=result)
            
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@main_router.get("/models", tags=["ML Models"])
async def get_models():
    """
    Получение списка доступных моделей
    
    Returns:
        Список моделей с информацией
    """
    try:
        models = ml_service.model_manager.get_available_models()
        current_model = ml_service.model_manager.get_current_model_info()
        
        return JSONResponse(content={
            "models": models,
            "current_model": current_model,
            "total_models": len(models)
        })
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@main_router.get("/tasks/{task_id}", tags=["ML Tasks"])
async def get_task_status(task_id: str, return_image: bool = False):
    """
    Получение статуса задачи

    Args:
        task_id: ID задачи
        return_image: если True и задача завершена с изображением, возвращает изображение

    Returns:
        Статус задачи и результат если готово, или изображение если return_image=True
    """
    try:
        status = ml_queue_integration.get_task_status(task_id)
        logger.info(f"Getting task status for task {task_id}: ")
        if not status:
            raise HTTPException(status_code=404, detail={"error": f"Task {task_id} not found"})

        # Если запрошено изображение и задача завершена
        if return_image and status.get("status") == "completed":
            result = status.get("result")
            if result:
                # Проверяем наличие изображения в разных форматах
                image_bytes = None

                # 1. Если есть image_bytes (байты)
                if "image_bytes" in result:
                    logger.info("Found image_bytes in result, returning it")
                    image_bytes = result["image_bytes"]
                    # Удаляем из результата, чтобы не дублировать в JSON
                    result.pop("image_bytes")
                # 2. Если есть annotated_image_base64 (на практике может быть hex или base64)
                elif "annotated_image_base64" in result:
                    logger.info("Found annotated_image_base64 in result, converting to bytes")
                    raw_str = result.pop("annotated_image_base64")

                    try:
                        s = raw_str
                        # Нормализуем к str
                        if isinstance(s, (bytes, bytearray)):
                            # если вдруг прилетели байты, пробуем трактовать как ascii-строку с hex/base64
                            s = bytes(s).decode("ascii", errors="strict")
                        else:
                            s = str(s)

                        s = "".join(s.strip().split())

                        is_hex = (
                                len(s) % 2 == 0
                                and all(c in "0123456789abcdefABCDEF" for c in s)
                        )

                        if is_hex:
                            image_bytes = bytes.fromhex(s)
                        else:
                            # data URL: data:image/jpeg;base64,....
                            if "," in s and "base64" in s.split(",", 1)[0]:
                                s = s.split(",", 1)[1]
                            pad_len = (-len(s)) % 4
                            if pad_len:
                                s += "=" * pad_len
                            image_bytes = base64.b64decode(s, validate=False)

                        logger.info(f"Decoded annotated image, type={type(image_bytes)!r}, length={len(image_bytes)}")
                    except Exception as e:
                        logger.error(f"Failed to decode annotated_image_base64 to bytes: {e}")
                        image_bytes = None

                if image_bytes:
                    # Возвращаем изображение
                    detection_meta = {
                        "task_id": task_id,
                        "status": status.get("status"),
                        "image_id": result.get("image_id", ""),
                        "total_objects": result.get("total_objects", 0),
                        "detections": result.get("detections", []),
                    }
                    return Response(
                        content=bytes(image_bytes),
                        media_type="image/jpeg",
                        headers={
                            "Access-Control-Allow-Origin": "*",
                            "Access-Control-Expose-Headers": "X-Detection-Meta, X-Task-ID, X-Total-Objects",
                            "X-Detection-Meta": json.dumps(detection_meta, ensure_ascii=False),
                            "X-Task-ID": detection_meta["image_id"],
                            "X-Total-Objects": str(detection_meta["total_objects"]),
                        }
                    )


        # Возвращаем JSON с статусом
        return JSONResponse(content=status)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@main_router.post("/switch-model", tags=["ML Models"])
async def switch_model(request: SwitchModelRequest):
    """
    Переключение активной модели
    
    Args:
        request: запрос с именем модели
    
    Returns:
        Результат переключения
    """
    try:
        model_name = request.model_name
        
        success = ml_service.model_manager.switch_model(model_name)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to switch to model {model_name}"
            )
        
        current_model = ml_service.model_manager.get_current_model_info()
        
        return JSONResponse(content={
            "message": f"Switched to model {model_name}",
            "current_model": current_model,
            "success": True
        })
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@main_router.get("/queue/stats", tags=["ML Queue"])
async def get_queue_stats():
    """
    Получение статистики очереди
    
    Returns:
        Статистика очереди
    """
    try:
        stats = ml_queue_integration.get_queue_stats()
        return JSONResponse(content=stats)
    except Exception as e:
        logger.error(f"Error getting queue stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@main_router.get("/queue/tasks", tags=["ML Queue"])
async def get_all_tasks():
    """
    Получение всех задач в очереди (для отладки)
    
    Returns:
        Список всех задач
    """
    try:
        from src.queue_manager import queue_manager
        task_queue = queue_manager.get_queue("ml_predictions")
        if not task_queue:
            return JSONResponse(content={"tasks": []})
        
        with task_queue.lock:
            tasks = [task.to_dict() for task in task_queue.tasks.values()]
        
        return JSONResponse(content={"tasks": tasks})
    except Exception as e:
        logger.error(f"Error getting all tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@main_router.post("/queue/cleanup", tags=["ML Queue"])
async def cleanup_queue():
    """
    Очистка старых задач из очереди
    
    Returns:
        Результат очистки
    """
    try:
        ml_queue_integration.cleanup_old_tasks(max_age_hours=1)
        return JSONResponse(content={
            "message": "Old tasks cleaned up",
            "success": True
        })
    except Exception as e:
        logger.error(f"Error cleaning up queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@main_router.get("/health", tags=["Health"])
async def health_check():
    """
    Проверка здоровья приложения
    
    Returns:
        Статус приложения
    """
    try:
        # Проверяем доступность ML сервиса
        models_available = len(ml_service.model_manager.get_available_models()) > 0
        
        # Проверяем очередь
        queue_stats = ml_queue_integration.get_queue_stats()
        queue_healthy = "queue_name" in queue_stats
        
        return JSONResponse(content={
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "ml_service": {
                "available": models_available,
                "current_model": ml_service.model_manager.current_model_name
            },
            "queue": {
                "healthy": queue_healthy,
                "stats": queue_stats
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Импортируем datetime для health check
from datetime import datetime