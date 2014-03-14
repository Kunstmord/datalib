from distutils.core import setup

setup(name='kagglelib',
      version='0.1',
      description='Library for data analysis - extracting, storing and retrieving features',
      author='George Oblapenko',
      author_email='kunstmord@kunstmord.com',
      url='https://github.com/Kunstmord/kagglelib',
      license="GPL",
      packages=['kagglelib'],
      package_dir={'kagglelib': 'src'},
      requires=['numpy', 'sqlalchemy']
      )