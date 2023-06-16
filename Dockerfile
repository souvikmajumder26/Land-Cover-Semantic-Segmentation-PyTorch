# set base image (host OS)
FROM python:3.9

# set the working directory in the container
WORKDIR /segment_project

# copy the dependencies file to the working directory
COPY requirements.txt .

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copy the content from the local directory to the working directory
COPY ./config ./config
COPY ./data ./data
COPY ./models ./models
COPY ./src ./src

# command to run on container start
# comment and uncomment either of the following lines based on whether to train or test the model
CMD [ "python", "./src/test.py" ]
# CMD [ "python", "./src/train.py" ]