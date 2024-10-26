from abc import ABC, abstractmethod


class MechanicalSystem(ABC):
  @abstractmethod
  def M(self, q):
    pass

  @abstractmethod
  def C(self, q, dq):
    pass

  @abstractmethod
  def G(self, q):
    pass

  @abstractmethod
  def B(self, q):
    pass

  @abstractmethod
  def U(self, q):
    pass

  @abstractmethod
  def K(self, q, dq):
    pass
