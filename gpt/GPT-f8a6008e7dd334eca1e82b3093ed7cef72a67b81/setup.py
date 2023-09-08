from setuptools import setup

setup(
    name='gpt',
    version='0.0.9',
    description='ChatGPT API',
    url='https://github.com/FreedomIntelligence/GPT',
    author='Feng Jiang',
    author_email='jeffreyjiang@cuhk.edu.cn',
    license='BSD 2-clause',
    packages=['gpt'],
    package_data = { '': ['apikey.config']},
    zip_safe=False,
    include_package_data=True,
    install_requires=[
                        # 'mpi4py>=2.0',
                      'requests',                     
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
