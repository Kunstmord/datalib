__author__ = 'George Oblapenko'
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"
__status__ = "Development"

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, PickleType, Integer, MetaData
from sqlalchemy.ext.mutable import Mutable

Base = declarative_base()
metadata = MetaData()


class MutableDict(Mutable, dict):
    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, MutableDict):
            if isinstance(value, dict):
                return MutableDict(value)
            return Mutable.coerce(key, value)
        else:
            return value

    def __delitem(self, key):
        dict.__delitem__(self, key)
        self.changed()

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.changed()

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


class TrainSet(Base):
    __tablename__ = 'train set'
    id = Column(Integer, primary_key=True)
    real_id = Column(String(60))
    path = Column(String(120))
    labels = Column(MutableDict.as_mutable(PickleType))
    features = Column(MutableDict.as_mutable(PickleType))