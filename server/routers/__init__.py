from fastapi import APIRouter

from server.routers.llm_router import llm

router = APIRouter()

# 注册路由结构
router.include_router(llm)  # /api/system/tools/*
