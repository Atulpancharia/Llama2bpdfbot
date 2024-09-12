from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from src.database import get_db
from src.models.users import User

router = APIRouter(prefix="/auth")

class UserLogin(BaseModel):
    email: str
    password: str


class UserResponse(BaseModel):
    id: int
    email: str
    name: str


@router.post("/users")
async def create_user(
    name: str,
    email: str,
    password: str,
    db: Session = Depends(get_db),
):
    # hashed_password = hash_password(password)
    user = User(name=name, email=email, password=password)
    db.add(user)
    db.commit()
    return {"message": "User successfully created", "data": user}


@router.post("/login", response_model=UserResponse)
async def login(
    data: UserLogin,
    db: Session = Depends(get_db),
):
    print("email=====", data.email, data.password)
    user = db.query(User).where(User.email == data.email).first()
    # user = result.scalar_one_or_none()
    print("user.password====", user)
    if user.password != data.password:
        return {"message": "Invalid email or password"}

    return UserResponse(id=user.id, email=user.email, name=user.name)