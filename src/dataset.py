from src import trainset, testset, errors

__author__ = 'George Oblapenko'
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"
__status__ = "Development"

from os.path import join, isfile
from os import walk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import csv


def cutoff_filename(prefix, suffix, filename):
    if prefix is not '':
        if filename.startswith(prefix):
            filename = filename[len(prefix):]
    if suffix is not '':
        if filename.endswith(suffix):
            filename = filename[:-len(suffix)]
    return filename


def extract_feature_base(dbpath, set_object, extractor, force_extraction=False, *args):
    extractor_name = extractor.__name__
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    for i in session.query(set_object).order_by(set_object.id):
        if i.features is None:
            i.features = {extractor_name: extractor(i.path, *args)}
        else:
            if (extractor_name not in i.features) or force_extraction is True:
                i.features[extractor_name] = extractor(i.path, *args)
    session.commit()
    session.close()


def return_features_base(dbpath, set_object, names):
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    if names == 'all':
        for i in session.query(set_object).order_by(set_object.id):
            row_list = []
            for feature in i.features:
                row_list.append(i.features[feature])
            return_list.append(row_list[:])
    else:
        for i in session.query(set_object).order_by(set_object.id):
            row_list = []
            for feature in i.features:
                if feature in names:
                    row_list.append(i.features[feature])
            return_list.append(row_list[:])
    return return_list


def return_features_numpy_base(dbpath, set_object, points_amt, names):

    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(1)

    if names == 'all':
        columns_amt = 0
        for feature in tmp_object.features:
            if type(tmp_object.features[feature]) is np.ndarray:
                columns_amt += tmp_object.features[feature].shape[0]
            else:
                columns_amt += 1
        return_array = np.zeros([points_amt, columns_amt])
        for i in enumerate(session.query(set_object).order_by(set_object.id)):
            counter = 0
            for feature in i[1].features:
                feature_val = i[1].features[feature]
                if type(feature_val) is np.ndarray:
                    columns_amt = feature_val.shape[0]
                    return_array[i[0], counter:counter + columns_amt] = feature_val[:]
                    counter += feature_val.shape[0]
                else:
                    return_array[i[0], counter] = feature_val
                    counter += 1
    else:
        columns_amt = 0
        for feature in tmp_object.features:
            if feature in names:
                if type(tmp_object.features[feature]) is np.ndarray:
                    columns_amt += tmp_object.features[feature].shape[0]
                else:
                    columns_amt += 1
        return_array = np.zeros([points_amt, columns_amt])
        for i in enumerate(session.query(set_object).order_by(set_object.id)):
            counter = 0
            for feature in i[1].features:
                if feature in names:
                    feature_val = i[1].features[feature]
                    if type(feature_val) is np.ndarray:
                        columns_amt = feature_val.shape[0]
                        return_array[i[0], counter:counter + columns_amt] = feature_val[:]
                        counter += feature_val.shape[0]
                    else:
                        return_array[i[0], counter] = feature_val
                        counter += 1
    return return_array


class DataSetBase:
    def __init__(self, path_to_set, path_to_db, db_name):
        self.path_to_set = path_to_set

        dbpath = join(path_to_db, db_name)
        if isfile(dbpath):
            self._prepopulated = True
        else:
            self._prepopulated = False
        self.dbpath = dbpath


class UnlabeledDataSet(DataSetBase):
    def __init__(self, path_to_set, path_to_db, custom_name=None, file_prefix='', file_suffix=''):
        if custom_name is None:
            custom_name = 'test.db'
        DataSetBase.__init__(self, path_to_set, path_to_db, custom_name)
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.points_amt = 0
        if self._prepopulated is True:
            for (dirpath, dirnames, filenames) in walk(self.path_to_set):
                self.points_amt = len(filenames)

    def prepopulate(self):
        if self._prepopulated is False:
            engine = create_engine('sqlite:////' + self.dbpath)
            testset.Base.metadata.create_all(engine)
            self._prepopulated = True
            session_cl = sessionmaker(bind=engine)
            session = session_cl()

            for (dirpath, dirnames, filenames) in walk(self.path_to_set):
                for f_name in filenames:
                    datapoint = testset.TestSet(real_id=cutoff_filename(self.file_prefix, self.file_suffix, f_name),
                                                path=f_name, features=None)
                    session.add(datapoint)
                    self.points_amt += 1
            session.commit()
            session.close()

    def extract_feature(self, extractor, force_extraction=False, *args):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            extract_feature_base(self.dbpath, testset.TestSet, extractor, force_extraction, *args)

    def return_features(self, names='all'):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_base(self.dbpath, testset.TestSet, names)

    def return_features_numpy(self, names='all'):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            if self.points_amt == 0:
                raise errors.EmptyDatabase(self.dbpath, True)
            else:
                return return_features_numpy_base(self.dbpath, testset.TestSet, self.points_amt, names)


class LabeledDataSet(DataSetBase):
    def __init__(self, path_to_set, path_to_db, path_to_labels, delimiter=',', custom_name=None, label_dict=None,
                 label_header=True, file_prefix='', file_suffix=''):
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.label_header = label_header
        self.path_to_labels = path_to_labels
        self.delimiter = delimiter
        self.label_dict = label_dict
        self.points_amt = 0

        if custom_name is None:
            custom_name = 'train.db'
        DataSetBase.__init__(self, path_to_set, path_to_db, custom_name)
        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self.points_amt = 0
        if self._prepopulated is True:
            for (dirpath, dirnames, filenames) in walk(self.path_to_set):
                self.points_amt = len(filenames)

    def prepopulate(self):
        if self._prepopulated is False:
            engine = create_engine('sqlite:////' + self.dbpath)
            trainset.Base.metadata.create_all(engine)
            self._prepopulated = True
            session_cl = sessionmaker(bind=engine)
            session = session_cl()

            labels_csv = open(self.path_to_labels, 'r')
            reader = csv.reader(labels_csv, delimiter=self.delimiter)
            if self.label_header is True:
                labels = reader.next()
            for (dirpath, dirnames, filenames) in walk(self.path_to_set):
                for f_name in filenames:
                    labels = reader.next()
                    labels = labels[1:]  # strip first piece, it's the file ID - should probably check with filename?
                    writeable_labels = {'original': labels}
                    if self.label_dict is not None:
                        for label_transform in self.label_dict:
                            if label_transform in labels:
                                labels[labels.index(label_transform)] = self.label_dict[label_transform]
                    writeable_labels['transformed'] = labels
                    datapoint = trainset.TrainSet(real_id=cutoff_filename(self.file_prefix, self.file_suffix, f_name),
                                                  labels=writeable_labels, path=f_name, features=None)
                    session.add(datapoint)
                    self.points_amt += 1
            session.commit()
            session.close()

    def extract_feature(self, extractor, force_extraction=False, *args):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            extract_feature_base(self.dbpath, trainset.TrainSet, extractor, force_extraction, *args)

    def return_features(self, names='all'):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_base(self.dbpath, trainset.TrainSet, names)

    def return_features_numpy(self, names='all'):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            if self.points_amt == 0:
                raise errors.EmptyDatabase(self.dbpath, True)
            else:
                return return_features_numpy_base(self.dbpath, trainset.TrainSet, self.points_amt, names)

    def return_labels(self, original=False):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            engine = create_engine('sqlite:////' + self.dbpath)
            trainset.Base.metadata.create_all(engine)
            session_cl = sessionmaker(bind=engine)
            session = session_cl()
            return_list = []
            for i in session.query(trainset.TrainSet).order_by(trainset.TrainSet.id):
                if original is True:
                    row_list = i.labels['original']
                else:
                    row_list = i.labels['transformed']
                return_list.append(row_list[:])
            session.close()
            return return_list

    def return_labels_numpy(self, original=False):
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            engine = create_engine('sqlite:////' + self.dbpath)
            trainset.Base.metadata.create_all(engine)
            session_cl = sessionmaker(bind=engine)
            session = session_cl()
            tmp_object = session.query(trainset.TrainSet).get(1)

            columns_amt = len(tmp_object.labels['original'])
            return_array = np.zeros([self.points_amt, columns_amt])
            for i in enumerate(session.query(trainset.TrainSet).order_by(trainset.TrainSet.id)):
                if original is False:
                    return_array[i[0], :] = i[1].labels['transformed']
                else:
                    return_array[i[0], :] = i[1].labels['original']
            session.close()
            return return_array