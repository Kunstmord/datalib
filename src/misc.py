__author__ = 'georgeoblapenko'
__license__ = "GPL"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"

from sqlalchemy.ext.mutable import Mutable


class MutableDict(Mutable, dict):
    @classmethod
    def coerce(cls, key, value):
        if not isinstance(value, MutableDict):
            if isinstance(value, dict):
                return MutableDict(value)
            return Mutable.coerce(key, value)
        else:
            return value

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self.changed()

    def __setitem__(self, key, value):
        dict.__setitem__(self, key, value)
        self.changed()

    def __getstate__(self):
        return dict(self)

    def __setstate__(self, state):
        self.update(state)


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