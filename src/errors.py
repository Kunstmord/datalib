__author__ = 'George Oblapenko'
__license__ = "GPL"
__version__ = "0.1"
__maintainer__ = "George Oblapenko"
__email__ = "kunstmord@kunstmord.com"
__status__ = "Development"


class InsufficientData(Exception):
    def __init__(self, name, error_type, path=None):
        self.name = name
        self.error_type = error_type
        self.path = path

    def __str__(self):
        if self.path is None:
            return 'No ' + str(self.name) + ' ' + str(self.error_type)
        else:
            return 'No ' + str(self.name) + ' ' + str(self.error_type) + ' (at ' + str(self.path) + ')'


class EmptyDatabase(Exception):
    def __init__(self, path):
        self.path = path

    def __str__(self):
        return 'Database at ' + str(self.path) + ' is empty, run prepopulate() first'