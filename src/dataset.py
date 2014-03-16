__author__ = 'George Oblapenko'
__license__ = "GPL"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"

from os.path import join, isfile
from os import walk
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import numpy as np
import trainset
import testset
import errors
from misc import cutoff_filename


def extract_feature_base(dbpath, folder_path, set_object, extractor, force_extraction=False, verbose=0, *args):
    """
    Generic function which extracts a feature and stores it in the database

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    folder_path : string, path to folder where the files are stored
    set_object : object (either TestSet or TrainSet) which is stored in the database
    extractor : function, which takes the path of a data point and *args as parameters and returns a feature
    force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
    exists in the database, otherwise, will only extract if the feature doesn't exist in the database.
    default value: False
    verbose : int, if bigger than 0, will print the current number of the file for which data is being extracted
    ever verbose steps (for example, verbose=500 will print 0, 500, 1000 etc.). default value: 0
    *args : optional arguments for the extractor

    Returns
    -------
    None
    """
    extractor_name = extractor.__name__
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    a = 0

    tmp_object = session.query(set_object).get(1)
    if tmp_object.features is None:
        for i in session.query(set_object).order_by(set_object.id):
            i.features = {extractor_name: extractor(join(folder_path, i.path), *args)}
            if verbose > 0:
                if a % verbose == 0:
                    print a
            a += 1
    elif (extractor_name not in tmp_object.features) or force_extraction is True:
        for i in session.query(set_object).order_by(set_object.id):
            i.features[extractor_name] = extractor(join(folder_path, i.path), *args)
            if verbose > 0:
                if a % verbose == 0:
                    print a
            a += 1
    session.commit()
    session.close()
    return None


def extract_feature_dependent_feature_base(dbpath, folder_path, set_object, extractor, force_extraction=False,
                                           verbose=0, *args):
    """
    Generic function which extracts a feature which may be dependent on other features and stores it in the database

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    folder_path : string, path to folder where the files are stored
    set_object : object (either TestSet or TrainSet) which is stored in the database
    extractor : function, which takes the path of a data point, a dictionary of all other features and *args as
    parameters and returns a feature
    force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
    exists in the database, otherwise, will only extract if the feature doesn't exist in the database.
    default value: False
    verbose : int, if bigger than 0, will print the current number of the file for which data is being extracted
    ever verbose steps (for example, verbose=500 will print 0, 500, 1000 etc.). default value: 0
    *args : optional arguments for the extractor

    Returns
    -------
    None
    """
    extractor_name = extractor.__name__
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    a = 0

    tmp_object = session.query(set_object).get(1)
    if tmp_object.features is None:
        for i in session.query(set_object).order_by(set_object.id):
            i.features = {extractor_name: extractor(join(folder_path, i.path), None, *args)}
            if verbose > 0:
                if a % verbose == 0:
                    print a
            a += 1
    elif (extractor_name not in tmp_object.features) or force_extraction is True:
        for i in session.query(set_object).order_by(set_object.id):
            i.features[extractor_name] = extractor(join(folder_path, i.path), i.features, *args)
            if verbose > 0:
                if a % verbose == 0:
                    print a
            a += 1
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
    return_list : list of lists, each 'inside list' corresponds to a single data point, each element of the 'inside list' is a
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
    points_amt : int, number of data points in the database
    names : list of strings, a list of feature names which are to be retrieved from the database, if equal to 'all',
    all features will be returned

    Returns
    -------
    return_array : ndarray of features, each row corresponds to a single datapoint. If a single feature
    is a 1d numpy array, then it will be unrolled into the resulting array. Higher-dimensional numpy arrays are not
    supported.
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


def return_real_id_base(dbpath, set_object):
    """
    Generic function which returns a list of real_id's

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database

    Returns
    -------
    return_list : list of real_id values for the dataset (a real_id is the filename minus the suffix and prefix)
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    for i in session.query(set_object).order_by(set_object.id):
        return_list.append(i.real_id)
    return return_list


def return_feature_list_base(dbpath, set_object):
    """
    Generic function which returns a list of the names of all available features

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database

    Returns
    -------
    return_list : list of strings corresponding to all available features
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    tmp_object = session.query(set_object).get(1)
    for feature in tmp_object.features:
        return_list.append(feature)
    return return_list


def return_feature_list_numpy_base(dbpath, set_object):
    """
    Generic function which returns a list of tuples containing, each containing the name of the feature
    and the length of the corresponding 1d numpy array of the feature (or length of the list)

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database

    Returns
    -------
    return_list : list of tuples containing the name of the feature and the length of the corresponding list or
    1d numpy array
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    return_list = []
    tmp_object = session.query(set_object).get(1)
    for feature in tmp_object.features:
        if type(tmp_object.features[feature]) is np.ndarray:
            flength = tmp_object.features[feature].shape[0]
        else:
            flength = 1
        return_list.append((feature, flength))
    return return_list


def copy_features_base(dbpath_origin, dbpath_destination, set_object, force_copy=False):
    """
    Generic function which copies features from one database to another (base object should be of the same type)

    Parameters
    ----------
    dbpath_origin : string, path to SQLite database file from which the features will be copied
    dbpath_destination : string, path to SQLite database file to which the features will be copied
    set_object : object (either TestSet or TrainSet) which is stored in the database
    force_copy : boolean, if True - will overwrite features with same name when copying, if False, won't;
    default value: False

    Returns
    -------
    None
    """
    engine_origin = create_engine('sqlite:////' + dbpath_origin)
    engine_destination = create_engine('sqlite:////' + dbpath_destination)
    session_cl_origin = sessionmaker(bind=engine_origin)
    session_cl_destination = sessionmaker(bind=engine_destination)
    session_origin = session_cl_origin()
    session_destination = session_cl_destination()
    if force_copy is True:
        for i in session_origin.query(set_object).order_by(set_object.id):
            dest_obj = session_destination.query(set_object).get(i.id)
            for feature in i.features:
                if dest_obj.features is not None:
                    dest_obj.features[feature] = i.features[feature]
                else:
                    dest_obj.features = {feature: i.features[feature]}
    else:
        for i in session_origin.query(set_object).order_by(set_object.id):
            dest_obj = session_destination.query(set_object).get(i.id)
            for feature in i.features:
                if dest_obj.features is not None:
                    if (feature not in dest_obj.features) or force_copy is True:
                        dest_obj.features[feature] = i.features[feature]
                else:
                    dest_obj.features = {feature: i.features[feature]}
    return None


def return_single_real_id_base(dbpath, set_object, object_id):
    """
    Generic function which returns a real_id string of an object specified by the object_id

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database

    Returns
    -------
    real_id : string
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    return tmp_object.real_id


def return_single_path_base(dbpath, set_object, object_id):
    """
    Generic function which returns a path (path is relative to the path_to_set stored in the database) of an object
    specified by the object_id

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database

    Returns
    -------
    path : string
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    return tmp_object.path


def return_single_features_base(dbpath, set_object, object_id):
    """
    Generic function which returns the features of an object specified by the object_id

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database

    Returns
    -------
    features : dict containing the features
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    return tmp_object.features


def return_single_convert_numpy_base(dbpath, folder_path, set_object, object_id, converter, *args):
    """
    Generic function which converts an object specified by the object_id into a numpy array and returns the array,
    the conversion is done by the 'converter' function

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    folder_path : string, path to folder where the files are stored
    set_object : object (either TestSet or TrainSet) which is stored in the database
    object_id : int, id of object in database
    converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
    *args : optional arguments for the converter

    Returns
    -------
    result : ndarray
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(object_id)
    return converter(join(folder_path, tmp_object.path), *args)


def return_multiple_convert_numpy_base(dbpath, folder_path, set_object, start_id, end_id, converter, *args):
    """
    Generic function which converts several objects, with ids in the range (start_id, end_id)
    into a 2d numpy array and returns the array, the conversion is done by the 'converter' function

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    folder_path : string, path to folder where the files are stored
    set_object : object (either TestSet or TrainSet) which is stored in the database
    start_id : the id of the first object to be converted
    end_id : the id of the last object to be converted
    converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
    *args : optional arguments for the converter

    Returns
    -------
    result : 2-dimensional ndarray
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    tmp_object = session.query(set_object).get(start_id)
    converted = converter(join(folder_path, tmp_object.path), *args)
    columns_amt = converted.shape[0]
    return_array = np.zeros([end_id - start_id, columns_amt])
    for i in xrange(end_id - start_id + 1):
        tmp_object = session.query(set_object).get(start_id + i)
        return_array[i, :] = converter(join(folder_path, tmp_object.path), *args)
    return return_array


def dump_feature_base(dbpath, set_object, points_amt, feature_name, feature, force_extraction=True):
    """
    Generic function which dumps a list of lists or ndarray of features into database (allows to
    copy features from a pre-existing .txt/.csv/.whatever file, for example)

    Parameters
    ----------
    dbpath : string, path to SQLite database file
    set_object : object (either TestSet or TrainSet) which is stored in the database
    points_amt : int, number of data points in the database
    feature : list of lists or ndarray, contains the data to be written to the database
    force_extraction : boolean, if True - will overwrite any existing feature with this name
    default value: False

    Returns
    -------
    None
    """
    engine = create_engine('sqlite:////' + dbpath)
    session_cl = sessionmaker(bind=engine)
    session = session_cl()
    a = 0

    tmp_object = session.query(set_object).get(1)
    if type(feature) is np.ndarray:
        if feature.shape[0] != points_amt:
            raise errors.WrongSize(feature_name)
        else:
            if tmp_object.features is None:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features = {feature_name: feature_name[a, :]}
                    a += 1
            elif (feature_name not in tmp_object.features) or force_extraction is True:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features[feature_name] = feature_name[a, :]
                    a += 1
    else:
        if len(feature) != points_amt:
            raise errors.WrongSize(feature_name)
        else:
            if tmp_object.features is None:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features = {feature_name: feature_name[a]}
                    a += 1
            elif (feature_name not in tmp_object.features) or force_extraction is True:
                for i in session.query(set_object).order_by(set_object.id):
                    i.features[feature_name] = feature_name[a]
                    a += 1
    session.commit()
    session.close()
    return None


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
    set_object : object (either TestSet or TrainSet) which is stored in the database
    db_base : declarative_base, used to create the engine (either testset.Base or trainset.Base)
    path_to_set : string, path to the folder containing the data point files
    path_to_db : string, path to the folder where the SQLite database is stored
    db_name : string, name of the SQLite database file
    file_prefix : string to cut off from start of filename when creating the 'real_id' field for data point
    file_suffix : string to cut off from end of filename when creating the 'real_id' field for data point
    """
    def __init__(self, set_object, db_base, path_to_set, path_to_db, db_name, file_prefix, file_suffix):
        self.path_to_set = path_to_set

        self.file_prefix = file_prefix
        self.file_suffix = file_suffix
        self._set_object = set_object
        self._db_base = db_base
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
            self._db_base.metadata.create_all(engine)
            self._prepopulated = True
            session_cl = sessionmaker(bind=engine)
            session = session_cl()

            for (dirpath, dirnames, filenames) in walk(self.path_to_set):
                for f_name in filenames:
                    datapoint = self._set_object(real_id=cutoff_filename(self.file_prefix, self.file_suffix, f_name),
                                                 path=f_name, features=None)
                    session.add(datapoint)
                    self.points_amt += 1
            session.commit()
            session.close()
        return None

    def extract_feature(self, extractor, force_extraction=False, verbose=0, *args):
        """
        Extracts a feature and stores it in the database

        Parameters
        ----------
        extractor : function, which takes the path of a data point and *args as parameters and returns a feature
        force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
        exists in the database, otherwise, will only extract if the feature doesn't exist in the database.
        default value: False
        verbose : int, if bigger than 0, will print the current number of the file for which data is being extracted
        *args : optional arguments for the extractor

        Returns
        -------
        None
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return extract_feature_base(self.dbpath, self.path_to_set, self._set_object, extractor, force_extraction,
                                        verbose, *args)

    def extract_feature_dependent_feature(self, extractor, force_extraction=False, verbose=0, *args):
        """
        Extracts a feature which may be dependent on other features and stores it in the database

        Parameters
        ----------
        extractor : function, which takes the path of a data point, a dictionary of all other features and *args as
        parameters and returns a feature
        force_extraction : boolean, if True - will re-extract feature even if a feature with this name already
        exists in the database, otherwise, will only extract if the feature doesn't exist in the database.
        default value: False
        verbose : int, if bigger than 0, will print the current number of the file for which data is being extracted
        *args : optional arguments for the extractor

        Returns
        -------
        None
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return extract_feature_dependent_feature_base(self.dbpath, self.path_to_set, self._set_object, extractor,
                                                          force_extraction, verbose, *args)

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
            return return_features_base(self.dbpath, self._set_object, names)

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
            return return_features_numpy_base(self.dbpath, self._set_object, self.points_amt, names)

    def return_real_id(self):
        """
        Returns a list of real_id's

        Parameters
        ----------

        Returns
        -------
        A list of real_id values for the dataset (a real_id is the filename minus the suffix and prefix)
        """
        if self._prepopulated is False:
            raise errors.EmptyDatabase(self.dbpath)
        else:
            return return_real_id_base(self.dbpath, self._set_object)

    def return_feature_list(self):
        """
        Returns a list of the names of all available features

        Parameters
        ----------

        Returns
        -------
        A list of strings corresponding to all available features
        """
        return return_feature_list_base(self.dbpath, self._set_object)

    def return_feature_list_numpy(self):
        """
        Returns a list of tuples containing, each containing the name of the feature and the length of the
        corresponding 1d numpy array of the feature (or length of the list)

        Parameters
        ----------

        Returns
        -------
        A list of tuples containing the name of the feature and the length of the corresponding list or 1d numpy array
        """
        return return_feature_list_numpy_base(self.dbpath, self._set_object)

    def copy_features(self, dbpath_origin, force_copy=False):
        """
        Copies features from one database to another (base object should be of the same type)

        Parameters
        ----------
        dbpath_origin : string, path to SQLite database file from which the features will be copied
        force_copy : boolean, if True - will overwrite features with same name when copying, if False, won't;
        default value: False

        Returns
        -------
        None
        """
        copy_features_base(dbpath_origin, self.dbpath, self._set_object, force_copy)
        return None

    def return_single_real_id(self, object_id):
        """
        Returns a real_id string of an object specified by the object_id

        Parameters
        ----------
        object_id : int, id of object in database

        Returns
        -------
        real_id : string
        """
        return return_single_real_id_base(self.dbpath, self._set_object, object_id)

    def return_single_path_base(self, object_id):
        """
        Returns a path (path is relative to the path_to_set stored in the database) of an object
        specified by the object_id

        Parameters
        ----------
        object_id : int, id of object in database

        Returns
        -------
        path : string
        """
        return return_single_path_base(self.dbpath, self._set_object, object_id)

    def return_single_features(self, object_id):
        """
        Returns the features of an object specified by the object_id

        Parameters
        ----------
        object_id : int, id of object in database

        Returns
        -------
        features : dict containing the features
        """
        return return_single_features_base(self.dbpath, self._set_object, object_id)

    def return_single_convert_numpy(self, object_id, converter, *args):
        """
        Converts an object specified by the object_id into a numpy array and returns the array,
        the conversion is done by the 'converter' function

        Parameters
        ----------
        object_id : int, id of object in database
        converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
        *args : optional arguments for the converter

        Returns
        -------
        result : ndarray
        """
        return return_single_convert_numpy_base(self.dbpath, self.path_to_set, self._set_object, object_id, converter,
                                                *args)

    def return_multiple_convert_numpy(self, start_id, end_id, converter, *args):
        """
        Converts several objects, with ids in the range (start_id, end_id)
        into a 2d numpy array and returns the array, the conversion is done by the 'converter' function

        Parameters
        ----------
        start_id : the id of the first object to be converted
        end_id : the id of the last object to be converted, if equal to -1, will convert all data points in range
        (start_id, <id of last element in database>)
        converter : function, which takes the path of a data point and *args as parameters and returns a numpy array
        *args : optional arguments for the converter

        Returns
        -------
        result : 2-dimensional ndarray
        """
        if end_id == -1:
            end_id = self.points_amt
        return return_multiple_convert_numpy_base(self.dbpath, self.path_to_set, self._set_object, start_id, end_id,
                                                  converter, *args)

    def dump_feature(self, feature_name, feature, force_extraction=True):
        """
        Dumps a list of lists or ndarray of features into database (allows to
        copy features from a pre-existing .txt/.csv/.whatever file, for example)

        Parameters
        ----------
        feature : list of lists or ndarray, contains the data to be written to the database
        force_extraction : boolean, if True - will overwrite any existing feature with this name
        default value: False

        Returns
        -------
        None
        """
        dump_feature_base(self.dbpath, self._set_object, self.points_amt, feature_name, feature, force_extraction)
        return None


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
        DataSetBase.__init__(self, testset.TestSet, testset.Base, path_to_set, path_to_db, custom_name, file_prefix,
                             file_suffix)


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

        DataSetBase.__init__(self, trainset.TrainSet, trainset.Base, path_to_set, path_to_db, custom_name, file_prefix,
                             file_suffix)

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

    def return_single_labels(self, object_id):
        """
        Returns all labels for an object specified by the object_id

        Parameters
        ----------
        object_id : int, id of object in database

        Returns
        -------
        result : list of labels
        """
        engine = create_engine('sqlite:////' + self.dbpath)
        trainset.Base.metadata.create_all(engine)
        session_cl = sessionmaker(bind=engine)
        session = session_cl()
        tmp_object = session.query(trainset.TrainSet).get(object_id)
        return tmp_object.labels