from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.storage.postgres.manager import pg_manager
from src.utils import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan事件管理器"""
    # 初始化数据库连接
    # try:
    #     pg_manager.initialize()
    #     await pg_manager.create_business_tables()
    # except Exception as e:
    #     logger.error(f"Failed to initialize database during startup: {e}")


    yield
    
    # await pg_manager.close()
