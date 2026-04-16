import re

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.postgres.manager import pg_manager
from server.utils.auth_utils import AuthUtils

# 定义OAuth2密码承载器，指定token URL
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/token", auto_error=False)

# 公开路径列表，无需登录即可访问
PUBLIC_PATHS = [
    r"^/api/auth/token$",  # 登录
    r"^/api/auth/check-first-run$",  # 检查是否首次运行
    r"^/api/auth/initialize$",  # 初始化系统
    r"^/api$",  # Health Check
    r"^/api/system/health$",  # Health Check
    r"^/api/system/info$",  # 获取系统信息配置
]


# 获取数据库会话（异步版本）
async def get_db():
    async with pg_manager.get_async_session_context() as db:
        yield db



# 检查路径是否为公开路径
def is_public_path(path: str) -> bool:
    path = path.rstrip("/")  # 去除尾部斜杠以便于匹配
    for pattern in PUBLIC_PATHS:
        if re.match(pattern, path):
            return True
    return False
