import abc
import typing


class SensorBase(abc.ABC):
    @abc.abstractmethod
    def simulate(self, base_prim: str) -> typing.Any:
        ...

    @abc.abstractmethod
    def publish(self, base_topic: str) -> typing.Any:
        ...
