"""
Provides an SQLAlchemy class for an unlabeled dataset
"""
__author__ = 'George Oblapenko'
__license__ = "GPL"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, String, PickleType, Integer, MetaData
from misc import MutableDict

Base = declarative_base()
metadata = MetaData()


class TestSet(Base):
    """
    SQLAlchemy class for an unlabeled dataset
    """
    __tablename__ = 'test set'
    id = Column(Integer, primary_key=True)
    real_id = Column(String(60))
    path = Column(String(120))
    features = Column(MutableDict.as_mutable(PickleType))