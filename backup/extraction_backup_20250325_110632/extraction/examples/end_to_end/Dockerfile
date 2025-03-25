FROM python:3.9-slim

WORKDIR /app

# Install dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install markdown

# Create directories
RUN mkdir -p /app/test_results /app/test_results_dashboard

# Copy validation files
COPY validation.py /app/
COPY convert_for_validation.py /app/
COPY detect_doc_type.py /app/
COPY generate_validation_report.py /app/
COPY test_validation_framework.py /app/
COPY validate_all_tests.py /app/
COPY docker-validation.sh /app/
COPY simple_server.py /app/

# Copy expected format templates
COPY *expected_format*.json /app/
COPY *.md /app/

# Make scripts executable
RUN chmod +x /app/docker-validation.sh /app/simple_server.py

# Run validation and start server
CMD ["/bin/bash", "-c", "./docker-validation.sh && python ./simple_server.py"]

# Expose the port
EXPOSE 8000