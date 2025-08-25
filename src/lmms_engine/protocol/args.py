from typing import Any, Dict

from pydantic import BaseModel


class Args(BaseModel):
    extra_kwargs: Dict[str, Any] = None

    def __post_init__(self):
        if self.extra_kwargs is None:
            self.extra_kwargs = {}

    def to_dict(self):
        return self.model_dump()

    def to_json(self):
        return self.model_dump_json()
