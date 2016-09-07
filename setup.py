"""
Flask-SVM
---------

This is the description for that library
"""
from setuptools import setup


setup(
    name='Flask-SVM',
    version='0.1',
    url='http://github.com/mxamin/flask-svm/',
    license='BSD',
    author='mxamin',
    author_email='amin.solhizadeh@gmail.com',
    description='SVM extension for Flask',
    long_description=__doc__,
    py_modules=['flask_svm'],
    zip_safe=False,
    include_package_data=True,
    platforms='any',
    install_requires=[
        'Flask>=0.11.1',
        'scikit-learn>=0.17.1',
        'numpy>=1.11.1',
        'scipy>=0.18.0',
    ],
    classifiers=[
        'Environment :: Web Environment',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Internet :: WWW/HTTP :: Dynamic Content',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ]
)
