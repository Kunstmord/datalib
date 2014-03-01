from src import trainset, testset, errors

__author__ = 'George Oblapenko'
__license__ = "GPL"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"
__status__ = "Development"

from os.path import join, isfile
from os import walk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import csv


def cutoff_filename(prefix, suffix, input_str):
    """
    Cuts off the start and end of a string, as specified by 2 parameters

    Parameters
    ----------
    prefix : string, if input_str starts with prefix, will cut off prefix
    suffix : string, if input_str end with suffix, will cut off suffix
    input_str : the string to be processed

    Returns
    -------
    A string, from which the start and end have been cut
    """
    if prefix is not '':
        if input_str.startswith(prefix):
            input_str = input_str[len(prefix):]
    if suffix is not '':
        if input_str.endswith(suffix):
            input_str = input_str[:-len(suffix)]
    return input_str


def extract_feature_base(dbpath, folder_path, set_object, extractor, force_extraction=False, *args):
    """
    Generic function which extracts a feature and stores it in the database

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    folder_path : string, path to folder where the files are stored
    set_object : object (either TestSet or TrainSet) which is stored in the database
    extractor : function, which takes the path of a data point and *args as parameters and returns a feature
    force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
    exists in the database, otherwise, will only extract a new feature
    *args : optional arguments for the extractor

    Returns
    -------
    None
    """
    extractor_name = extractor.__name__
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    for i in session.query(set_object).order_by(set_object.id):
        if i.features is None:
            i.features = {extractor_name: extractor(join(folder_path, i.path), *args)}
        else:
            if (extractor_name not in i.features) or force_extraction is True:
                i.features[extractor_name] = extractor(join(folder_path, i.path), *args)
    session.commit()
    session.close()
    return None


def return_features_base(dbpath, set_object, names):
    """
    Generic function which returns a list of extracted features from the database

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
    all features will be returned

    Returns
    -------
    A list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside list' is a
    feature (can be of any type)
    """
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
    """
    Generic function which returns a 2d numpy array of extracted features

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    points_amt : number of data points in the database
    names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
    all features will be returned

    Returns
    -------
    A numpy array of features, each row corresponds to a single datapoint. If a single feature is a 1d numpy array,
    then it will be unrolled into the resulting array. Higher-dimensional numpy arrays are not supported.
    """
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
    """
    Generic class for a data set. Assumes that each data point is a separate file in the same directory.
    Saves some basic data, checks whether a database already exists, if it does, calculates the amount of
    data points in it.
    file_prefix and file_suffix are used to convert the file name into an id (in case one might need one):
    for example, if:
    file_prefix = ''
    file_suffix = '.jpg'
    then the file '12345.jpg' will have a 'real_id' of '12345' associated with it.

    Initialization parameters
    ----------
    path_to_set : string, path to the folder containing the data point files
    path_to_db : string, path to the folder where the SQLite database is stored
    db_name : string, name of the SQLite database file
    file_prefix : string to cut off from start of filename when creating the 'real_id' field for data point
    file_suffix : string to cut off from end of filename when creating the 'real_id' field for data point
    """
    def __init__(self, path_to_set, path_to_db, db_name, file_prefix, file_suffix):
        self.path_to_set = path_to_set

        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        dbpath = join(path_to_db, db_name)
        if isfile(dbpath):
            self._prepopulated = True
        else:
            self._prepopulated = False
        self.points_amt = 0
        if self._prepopulated is True:
            for (dirpath, dirnames, filenames) in walk(self.path_to_set):
                self.points_amt = len(filenames)
        self.dbpath = dbpath


class UnlabeledDataSet(DataSetBase):
    """
    A class for a data set where each data point has no label associated with it

    Initialization parameters
    ----------
    path_to_set : string, path to the folder containing the data point files
    path_to_db : string, path to the folder where the SQLite database is stored
    custom_name : string, optional, name of database file, default value: 'test.db'
    file_prefix : string, optional, string to cut off from start of filename when creating the 'real_id' field for
    data point, default value: ''
    file_suffix : string, optional, string to cut off from end of filename when creating the 'real_id' field for
    data point, default value: ''

    """
    def __init__(self, path_to_set, path_to_db, custom_name='test.db', file_prefix='', file_suffix=''):
        DataSetBase.__init__(self, path_to_set, path_to_db, custom_name, file_prefix, file_suffix)

    def prepopulate(self):
        """
        Creates a database file (if it doesn't exist, writes each data point's path, real_id into it)

        Parameters
        ----------
        self

        Returns
        -------
        None
        """
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
        return None

    def extract_feature(self, extractor, force_extraction=False, *args):
        """
        Extracts a feature and stores it in the database

        Parameters
        ----------
        extractor : function, which takes the path of a data point and *args as parameters and returns a feature
        force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
        exists in the database, otherwise, will only extract a new feature
        *args : optional arguments for the extractor

        Returns
        -------
        None
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return extract_feature_base(self.dbpath, self.path_to_set, testset.TestSet, extractor, force_extraction,
                                        *args)

    def return_features(self, names='all'):
        """
        Returns a list of extracted features from the database

        Parameters
        ----------
        names : list of strings, a list of feature names which are to be retrieved from the database, if equal
        to 'all', the all features will be returned, default value: 'all'

        Returns
        -------
        A list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside list' is a
        feature (can be of any type)
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_base(self.dbpath, testset.TestSet, names)

    def return_features_numpy(self, names='all'):
        """
        Returns a 2d numpy array of extracted features

        Parameters
        ----------
        names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
        all features will be returned, default value: 'all'

        Returns
        -------
        A numpy array of features, each row corresponds to a single datapoint. If a single feature is a 1d numpy array,
        then it will be unrolled into the resulting array. Higher-dimensional numpy arrays are not supported.
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_numpy_base(self.dbpath, testset.TestSet, self.points_amt, names)


class LabeledDataSet(DataSetBase):
    """
    A class for a data set where each data point has a label associated with it

    Initialization parameters
    ----------
    path_to_set : string, path to the folder containing the data point files
    path_to_db : string, path to the folder where the SQLite database is stored
    path_to_labels : string, path to the CSV file in which the labels are stored
    custom_name : string, optional, name of database file, default value: 'test.db'
    label_dict : dict, optional, defines a custom mapping of labels. Example: if label_dict is equal to
    {'a': 1}, then a label 'a' will be stored as 1 in the database (original labels are also stored)
    label_head : boolean, if True, the first row of the labels CSV file is a header row, if False, the first row
    already has some labels in it, default value: True
    file_prefix : string, optional, string to cut off from start of filename when creating the 'real_id' field for
    data point, default value: ''
    file_suffix : string, optional, string to cut off from end of filename when creating the 'real_id' field for
    data point, default value: ''

    """
    def __init__(self, path_to_set, path_to_db, path_to_labels, delimiter=',', custom_name='train.db', label_dict=None,
                 label_header=True, file_prefix='', file_suffix=''):

        self.label_header = label_header
        self.path_to_labels = path_to_labels
        self.delimiter = delimiter
        self.label_dict = label_dict

        DataSetBase.__init__(self, path_to_set, path_to_db, custom_name, file_prefix, file_suffix)

    def prepopulate(self):
        """
        Creates a database file (if it doesn't exist, writes each data point's path, real_id into it)

        Parameters
        ----------
        self

        Returns
        -------
        None
        """
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
        return None

    def extract_feature(self, extractor, force_extraction=False, *args):
        """
        Extracts a feature and stores it in the database

        Parameters
        ----------
        extractor : function, which takes the path of a data point and *args as parameters and returns a feature
        force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
        exists in the database, otherwise, will only extract a new feature
        *args : optional arguments for the extractor

        Returns
        -------
        None
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            extract_feature_base(self.dbpath, self.path_to_set, trainset.TrainSet, extractor, force_extraction, *args)
        return None

    def return_features(self, names='all'):
        """
        Returns a list of extracted features from the database

        Parameters
        ----------
        names : list of strings, a list of feature names which are to be retrieved from the database, if equal
        to 'all', the all features will be returned, default value: 'all'

        Returns
        -------
        A list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside list' is a
        feature (can be of any type)
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_base(self.dbpath, trainset.TrainSet, names)

    def return_features_numpy(self, names='all'):
        """
        Returns a 2d numpy array of extracted features

        Parameters
        ----------
        names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
        all features will be returned, default value: 'all'

        Returns
        -------
        A numpy array of features, each row corresponds to a single datapoint. If a single feature is a 1d numpy array,
        then it will be unrolled into the resulting array. Higher-dimensional numpy arrays are not supported.
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_features_numpy_base(self.dbpath, trainset.TrainSet, self.points_amt, names)

    def return_labels(self, original=False):
        """
        Returns the labels of the dataset

        Parameters
        ----------
        original : if True, will return original labels, if False, will return transformed labels (as defined by
        label_dict), default value: False

        Returns
        -------
        A list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside list' is a
        label
        """
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
        """
        Returns a 2d numpy array of labels

        Parameters
        ----------
        original : if True, will return original labels, if False, will return transformed labels (as defined by
        label_dict), default value: False

        Returns
        -------
        A numpy array of labels, each row corresponds to a single datapoint
        """
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