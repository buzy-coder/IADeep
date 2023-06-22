FROM python:3.6.13
RUN mkdir /workspace
ADD requirement.txt /workspace
ADD *.py /workspace/
WORKDIR /workspace
RUN pip install -r requirement.txt
# Deprecated, check https://pypi.org/project/sklearn/
# RUN pip install sklearn
RUN pip install scikit-learn
RUN pip install flask
RUN pip install more-itertools
ENV LOG_LEVEL=DEBUG

# CMD ["python", "tuner.py"]
