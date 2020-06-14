###################################################
# Dockerfile for running batch jobs with dask
###################################################
FROM daskdev/dask:latest

# Copy python scripts in working directory
RUN mkdir -p /scripts/
COPY ./Marchenko.py /scripts/Marchenko.py

# Start script
ENTRYPOINT ["tini", "-g", "--", "/usr/bin/prepare.sh"]