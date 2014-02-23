__author__ = 'George Oblapenko'
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"
__status__ = "Development"

from os.path import normcase, split, join, isfile
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import errors
import trainset
import testset


class DataSet:
    def __init__(self, path_to_set, path_to_db, train_set=False, path_to_labels=None, label_dict=None,
                 custom_name=None, file_prefix='', file_suffix=''):
        self._path_to_set = path_to_set
        self._label_dict = label_dict
        self._file_prefix = file_prefix
        self._file_suffix = file_suffix
        self._train_set = train_set

        if custom_name is None:
            if train_set is False:
                dbpath = join(path_to_db, 'test.db')
            else:
                dbpath = join(path_to_db, 'train.db')
        else:
            dbpath = join(path_to_db, custom_name)
        if isfile(dbpath):
            self._prepopulated = True
        else:
            self._prepopulated = False
        self._dbpath = dbpath

        if train_set is True and path_to_labels is None:
            raise errors.InsufficientData('labels', 'specified')
        if train_set is True and not isfile(path_to_labels):
            raise errors.InsufficientData('labels', 'found at specified path', path_to_labels)

    def prepopulate(self):
        if self._prepopulated is False:
            engine = create_engine('sqlite:////' + self._dbpath)
            if self._train_set:
                trainset.metadata.create_all(engine)
                # trainset.Base.create_all(engine)
            else:
                testset.Base.metadata.create_all(engine)
                # testset.metadata.create_all(engine)
                # testset.Base.create_all(engine)
            self._prepopulated = True

    def write_test(self):
        engine = create_engine('sqlite:////' + self._dbpath)
        session_cl = sessionmaker(bind=engine)
        session = session_cl()
        att1 = testset.TestSet(real_id='AA', path='vvv', features={'a': 23})
        session.add(att1)
        session.commit()
        session.close()

    def get_test(self):
        engine = create_engine('sqlite:////' + self._dbpath)
        session_cl = sessionmaker(bind=engine)
        session = session_cl()
        for instance in session.query(testset.TestSet).order_by(testset.TestSet.id):
            print instance.features

    def get_label_dict(self):
        return self._label_dict
