===========
datalib
===========

Provides a library which simplifies processing and extracting features (for machine learning) from files.
Stores features in a SQLite database, has label transformation options, functions which convert features to NumPy arrays,
etc.
Originally designed for use with the Galaxy Zoo challenge on Kaggle.
Requires numpy and SQLAlchemy.

===========
Why?
===========

This is an attempt to minimize the amount of effort needed to extract, save and retrieve new features and allow
a user to spend more time on more 'scientific' work. If several users are working on the same project,
they can each extract independent sets of features and then share the database files and copy the features they are
missing.

===========
Roadmap
===========

Add a pure SQL way of storing and retrieving features (by converting them to JSON format and creating columns via ALTER
TABLE statements), thus allowing easy use from other languages. At least add this functionality for numpy arrays
and lists.

Add a dataset class for CSV files (where all data is stored in a single CSV file).