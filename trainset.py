__author__ = 'George Oblapenko'
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"
__status__ = "Development"

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, PickleType, Integer, MetaData

Base = declarative_base()
metadata = MetaData()


class TrainSet(Base):
    __tablename__ = 'train set'
    id = Column(Integer, primary_key=True)
    real_id = Column(String(60))
    path = Column(String(120))
    labels = Column(PickleType)
    features = Column(PickleType)