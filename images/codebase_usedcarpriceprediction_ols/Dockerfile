# Use the official Python 3.11 base image
FROM python:3.11-slim

# Install necessary system dependencies
RUN apt-get update && apt-get install -y \
    python3.11-venv python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as the default version
RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# Install required Python packages
RUN pip install --no-cache-dir \
    pandas \
    joblib \
    statsmodels

COPY ../../code/Prediction_OLS.py /codeBase/

COPY images/codebase_usedcarpriceprediction_ols/ReadMe.md /codeBase/

# Set working directory
WORKDIR /tmp/codeBase