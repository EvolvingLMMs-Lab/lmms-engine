import pydantic
import json

class Test(pydantic.BaseModel):
    a: int = 1
    b: int = 2

print(json.dumps(Test()))
