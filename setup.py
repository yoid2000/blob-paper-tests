from setuptools import setup, find_packages

setup(
    name='blob_paper_tests',
    version='0.1',
    packages=find_packages(where='blob-paper-tests'),
    package_dir={'': 'blob-paper-tests'},
    install_requires=[],
    author='Paul Francis',
    author_email='francis@mpi-sws.org',
    description='The code for my blob paper',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: OS Independent',
    ],
    license='GNU General Public License v3',
)