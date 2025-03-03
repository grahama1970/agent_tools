"""Common constants for method validation."""

# Standard library packages that don't need analysis
STDLIB_PACKAGES = {
    'os', 'sys', 'json', 'time', 'datetime', 'math', 're', 'random',
    'collections', 'itertools', 'functools', 'typing', 'pathlib',
    'logging', 'tempfile', 'shutil', 'subprocess', 'io', 'contextlib',
    'threading', 'multiprocessing', 'asyncio', 'concurrent', 'socket',
    'ssl', 'http', 'urllib', 'email', 'xml', 'html', 'csv', 'sqlite3',
    'pickle', 'shelve', 'dbm', 'zlib', 'gzip', 'bz2', 'zipfile', 'tarfile',
    'hashlib', 'hmac', 'secrets', 'uuid', 'base64', 'binascii', 'struct',
    'codecs', 'unicodedata', 'stringprep', 'encodings', 'bisect', 'heapq',
    'array', 'weakref', 'types', 'copy', 'pprint', 'reprlib', 'enum',
    'numbers', 'cmath', 'decimal', 'fractions', 'statistics', 'unittest',
    'doctest', 'argparse', 'optparse', 'getopt', 'fileinput', 'stat',
    'filecmp', 'fnmatch', 'glob', 'linecache', 'shlex', 'configparser',
    'netrc', 'xdrlib', 'plistlib', 'ast', 'symtable', 'token', 'keyword',
    'tokenize', 'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis',
    'pickletools', 'distutils', 'ensurepip', 'venv', 'zipapp', 'platform',
    'errno', 'ctypes', 'gc', 'inspect', 'site', 'code', 'codeop', 'zipimport',
    'pkgutil', 'modulefinder', 'runpy', 'importlib', 'parser', 'abc',
    'atexit', 'traceback', 'warnings', 'dataclasses', 'contextvars',
    'builtins', '_thread', '_dummy_thread', '_dummy_threading'
}

# Common utility packages that don't need analysis
COMMON_UTIL_PACKAGES = {
    'requests', 'urllib3', 'aiohttp', 'httpx', 'pydantic', 'sqlalchemy',
    'alembic', 'fastapi', 'flask', 'django', 'pytest', 'unittest', 'nose',
    'numpy', 'pandas', 'scipy', 'matplotlib', 'seaborn', 'plotly', 'bokeh',
    'scikit-learn', 'tensorflow', 'torch', 'keras', 'pillow', 'opencv-python',
    'beautifulsoup4', 'lxml', 'html5lib', 'selenium', 'scrapy', 'celery',
    'redis', 'pymongo', 'psycopg2', 'mysqlclient', 'boto3', 'google-cloud',
    'azure-storage', 'paramiko', 'fabric', 'ansible', 'docker', 'kubernetes',
    'click', 'typer', 'rich', 'tqdm', 'colorama', 'termcolor', 'pyyaml',
    'toml', 'configparser', 'python-dotenv', 'cryptography', 'bcrypt',
    'passlib', 'jwt', 'marshmallow', 'cerberus', 'voluptuous', 'jsonschema',
    'graphene', 'strawberry-graphql', 'grpcio', 'protobuf', 'thrift',
    'zeromq', 'pika', 'kafka-python', 'confluent-kafka', 'elasticsearch',
    'loguru', 'structlog', 'sentry-sdk', 'prometheus-client', 'statsd',
    'datadog', 'newrelic', 'gunicorn', 'uvicorn', 'hypercorn', 'daphne',
    'supervisor', 'circus', 'watchdog', 'apscheduler', 'schedule',
    'python-crontab', 'python-dateutil', 'pytz', 'arrow', 'pendulum',
    'freezegun', 'faker', 'factory-boy', 'mimesis', 'hypothesis',
    'coverage', 'pytest-cov', 'mypy', 'pylint', 'flake8', 'black',
    'isort', 'autopep8', 'yapf', 'bandit', 'safety', 'poetry',
    'pipenv', 'virtualenv', 'tox', 'sphinx', 'mkdocs', 'pdoc'
}

__all__ = ['STDLIB_PACKAGES', 'COMMON_UTIL_PACKAGES'] 