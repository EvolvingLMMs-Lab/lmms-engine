from typing import Any, Dict

from pydantic import BaseModel


class Args(BaseModel):
<<<<<<< HEAD
    extra_kwargs: Dict[str, Any] | None = None

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}
=======
    extra_kwargs: Dict[str, Any] = {}
>>>>>>> origin/main

    def to_dict(self):
        return self.model_dump()

    def to_json(self):
        return self.model_dump_json()
