FROM jinaai/jina:2-py37-perf

# setup the workspace
COPY . /workspace
WORKDIR /workspace

# install the third-party requirements
RUN pip install --default-timeout=1000 --compile --no-cache-dir \
     -r requirements.txt

ENTRYPOINT ["jina", "executor", "--uses", "config.yml"]
