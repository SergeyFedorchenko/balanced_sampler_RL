FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy task files
COPY task.py .
COPY run_eval.py .
COPY test_grader.py .
COPY README.md .

# Default command: run grader validation
CMD ["python", "test_grader.py"]

