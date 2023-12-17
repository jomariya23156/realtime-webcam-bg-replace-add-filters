from pydantic import BaseModel

class Object(BaseModel):
    box: tuple[float, float, float, float]
    label: str

class Objects(BaseModel):
    objects: list[Object]