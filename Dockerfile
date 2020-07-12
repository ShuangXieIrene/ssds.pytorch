FROM nvcr.io/nvidia/pytorch:20.06-py3

RUN pip install opencv-python \
                pynvml \
                git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI

COPY . ssds.pytorch/
RUN pip install --no-cache-dir -e ssds.pytorch/