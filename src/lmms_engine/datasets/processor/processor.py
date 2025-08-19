from abc import ABC, abstractmethod

# I'm not very sure if we really need this class, since I found that pure_text_processor don't have a process method. Maybe we need to force all the processors to have a process method?
class Processor(ABC):
    @abstractmethod
    def process(self, *args, **kwargs) -> dict:
        raise NotImplementedError("Processor.process() is not implemented")
