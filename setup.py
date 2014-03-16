from distutils.core import setup

setup(name='mldatalib',
      version='0.1',
      description='Library for data analysis - extracting, storing and retrieving features',
      author='George Oblapenko',
      author_email='kunstmord@kunstmord.com',
      url='https://github.com/Kunstmord/datalib',
      license="GPL",
      packages=['mldatalib'],
      package_dir={'mldatalib': 'src'},
      requires=['numpy', 'sqlalchemy']
      )