# Step 1: Use the same Python version as your virtual environment
FROM python:3.10.6-buster

# Step 2: Set the working directory inside the container
WORKDIR Sci_papers

# Step 3: Copy the taxifare project files into the container
COPY api-framework api-framework
# COPY model model
# add file containing model weights

# Step 4: Copy dependencies (requirements.txt) into the container
COPY requirements_prod.txt requirements_prod.txt

# Step 5: Install dependencies
RUN pip install -r requirements_prod.txt
# only put --no-cache-dir for final image !!

# Step 6: Expose the port (FastAPI runs on port 8000 by default)
#EXPOSE 8000

# Step 7: Set the entrypoint to launch the FastAPI server
CMD uvicorn api-framework.api.fast:app --host 0.0.0.0 --port $PORT
