#!/bin/sh

echo Hello Ubicomp Challenge 2021 World!
gunicorn --chdir app app:app -w 1 --threads 1 --access-logfile=- --error-logfile=- --log-level=debug -b 0.0.0.0:80

exec "$@"
