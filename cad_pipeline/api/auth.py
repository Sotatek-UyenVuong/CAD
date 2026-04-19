"""api/auth.py — JWT authentication + RBAC for the CAD pipeline API.

Users collection schema (MongoDB):
  { _id: email, hashed_password, role: "admin"|"editor"|"viewer"|"user",
    created_at, last_login }

Endpoints:
  POST /auth/login          → { access_token, token_type, role, email }
  GET  /auth/me             → { email, role }
  POST /auth/register       → admin-only: create new user
  PATCH /auth/users/{email} → admin-only: change role / reset password

RBAC:
  user/viewer : full usage (upload, qa, search, create folders), no library delete
  editor      : same as user/viewer
  admin       : editor + DELETE files/folders + POST /auth/register + PATCH /auth/users

Usage in app.py:
  from cad_pipeline.api.auth import require_role, get_current_user
  @app.post("/upload")
  def upload(current_user=Depends(require_role("editor", "admin")), ...):
      ...
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from functools import wraps
from typing import Callable

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

from cad_pipeline.storage import mongo

# ── Config ─────────────────────────────────────────────────────────────────

SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "change-me-in-production-jwt-secret")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "480"))  # 8h

# ── Crypto ──────────────────────────────────────────────────────────────────

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── JWT ─────────────────────────────────────────────────────────────────────

def create_access_token(email: str, role: str) -> str:
    expire = datetime.now(tz=timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """Decode and validate JWT. Raises HTTPException on failure."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        role: str | None = payload.get("role")
        if not email or not role:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
            )
        return {"email": email, "role": role}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expired or invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Dependencies ────────────────────────────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> dict:
    """FastAPI dependency: decode token → return {email, role}."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def require_role(*allowed_roles: str) -> Callable:
    """Returns a FastAPI dependency that enforces role-based access.

    Usage:
        @app.delete("/files/{file_id}")
        def delete(user=Depends(require_role("admin"))):
            ...
    """
    def _dependency(
        credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
    ) -> dict:
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authorization header missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        user = decode_token(credentials.credentials)
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user['role']}' is not permitted. Required: {list(allowed_roles)}",
            )
        return user

    return _dependency


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str
    email: str


class RegisterRequest(BaseModel):
    email: str
    password: str
    role: str = "user"  # default role for new users


class UpdateUserRequest(BaseModel):
    role: str | None = None
    password: str | None = None


# ── Router ──────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """Authenticate user, return JWT access token.

    Returns 401 if email not found or password incorrect.
    """
    user = mongo.get_user(req.email)
    if not user or not verify_password(req.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )
    mongo.update_user_last_login(req.email)
    token = create_access_token(req.email, user["role"])
    return LoginResponse(
        access_token=token,
        role=user["role"],
        email=req.email,
    )


@router.get("/me")
def get_me(current_user: dict = Depends(get_current_user)):
    """Return the currently authenticated user's email and role."""
    return {"email": current_user["email"], "role": current_user["role"]}


@router.post("/signup", status_code=201, response_model=LoginResponse)
def signup(req: LoginRequest):
    """Public self-registration — anyone can create a user account.

    Returns a JWT immediately so the user is logged in right after sign-up.
    New accounts always receive the 'user' role; admins can upgrade later.
    """
    existing = mongo.get_user(req.email)
    if existing:
        raise HTTPException(status_code=409, detail="Email đã được sử dụng.")
    if len(req.password) < 6:
        raise HTTPException(status_code=422, detail="Mật khẩu phải ít nhất 6 ký tự.")
    mongo.upsert_user(
        email=req.email,
        hashed_password=hash_password(req.password),
        role="user",
    )
    token = create_access_token(req.email, "user")
    return LoginResponse(access_token=token, role="user", email=req.email)


@router.post("/register", status_code=201)
def register(
    req: RegisterRequest,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin-only: create a new user account with a specific role."""
    if req.role not in ("admin", "editor", "viewer", "user"):
        raise HTTPException(status_code=422, detail="role must be admin, editor, viewer, or user")
    existing = mongo.get_user(req.email)
    if existing:
        raise HTTPException(status_code=409, detail=f"User '{req.email}' already exists")
    mongo.upsert_user(
        email=req.email,
        hashed_password=hash_password(req.password),
        role=req.role,
    )
    return {"email": req.email, "role": req.role, "created": True}


@router.patch("/users/{email}")
def update_user(
    email: str,
    req: UpdateUserRequest,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin-only: change a user's role or reset their password."""
    user = mongo.get_user(email)
    if not user:
        raise HTTPException(status_code=404, detail=f"User '{email}' not found")
    updates: dict = {}
    if req.role is not None:
        if req.role not in ("admin", "editor", "viewer", "user"):
            raise HTTPException(status_code=422, detail="role must be admin, editor, viewer, or user")
        updates["role"] = req.role
    if req.password is not None:
        updates["hashed_password"] = hash_password(req.password)
    if updates:
        mongo.update_user(email, updates)
    return {"email": email, "updated": list(updates.keys())}


@router.get("/users")
def list_users(current_user: dict = Depends(require_role("admin"))):
    """Admin-only: list all user accounts (passwords omitted)."""
    users = mongo.list_users()
    return {"users": users}


@router.delete("/users/{email}", status_code=204)
def delete_user(
    email: str,
    current_user: dict = Depends(require_role("admin")),
):
    """Admin-only: delete a user account."""
    if email == current_user["email"]:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    deleted = mongo.delete_user(email)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"User '{email}' not found")


# ── Seed helper (call once to create first admin) ───────────────────────────

def seed_admin(email: str, password: str) -> None:
    """Create the initial admin account if it does not already exist.

    Call this from a management script or startup hook:
        from cad_pipeline.api.auth import seed_admin
        seed_admin("admin@example.com", "changeme")
    """
    if not mongo.get_user(email):
        mongo.upsert_user(
            email=email,
            hashed_password=hash_password(password),
            role="admin",
        )
        print(f"[auth] Seeded admin user: {email}")
    else:
        print(f"[auth] Admin user already exists: {email}")
