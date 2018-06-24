import sys
from setuptools import setup
from fresh._version import __version__

try:
    from setuptools_rust import RustExtension, Binding
except ImportError:
    import subprocess
    errno = subprocess.call([sys.executable, '-m', 'pip', 'install', 'setuptools-rust>=0.9.2'])
    if errno:
        print("Please install the 'setuptools-rust>=0.9.2' package")
        raise SystemExit(errno)
    else:
        from setuptools_rust import RustExtension, Binding


setup(name="fresh",
      version=__version__,
      author="Miles Granger",
      maintainer='Miles Granger',
      author_email='miles59923@gmail.com',
      maintainer_email='miles59923@gmail.com',
      keywords='deep learning machine python rust gpu',
      description='Automated, end-to-end deep learning',
      long_description='',
      packages=['fresh'],
      rust_extensions=[
              RustExtension('fresh.rust.example', 'Cargo.toml', binding=Binding.PyO3)
          ],
      license='BSD',
      url='https://github.com/milesgranger/fresh',
      zip_safe=False,
      setup_requires=['setuptools-rust>=0.9.2', 'pytest-runner'],
      install_requires=[],
      tests_require=['pytest'],
      test_suite='tests',
      include_package_data=True,
      classifiers=[
            'Development Status :: 3 - Alpha',
            'Intended Audience :: Developers',
            'Intended Audience :: Financial and Insurance Industry',
            'Intended Audience :: Information Technology',
            'Intended Audience :: Science/Research',
            'Programming Language :: Python',
            'Programming Language :: Python :: 3.5',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3 :: Only',
            'Programming Language :: Rust',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: POSIX',
            'Operating System :: Unix',
            'Operating System :: MacOS :: MacOS X',
      ],
      )