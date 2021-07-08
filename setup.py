from setuptools import find_packages, setup
setup(
    name='mlctlsriracha',
    version='0.0.1',
    description='A project for mlctl jobs to abstract boilerplate APIs',
    author='Alex Chung',
    author_email='alex@socialg.tech',
    url='https://github.com/awcchungster/sriracha',
    packages=find_packages(),
    install_requires=[
        'google-cloud-storage>=1.40.0'
    ],
)