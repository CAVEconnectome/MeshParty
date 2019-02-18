FROM python:3.7.2

#apt-get
RUN apt-get update && apt-get install --yes build-essential libgl1-mesa-dev mesa-utils libgl1-mesa-glx
RUN apt-get -y install cmake libboost-all-dev

#CYTHON AND JUPYTER
RUN pip install cython==0.29
RUN pip install jupyter
COPY jupyter_notebook_config.py /root/.jupyter/


#ASSIMP
WORKDIR /usr/local/assimp
RUN apt-get update -y
RUN apt-get update
RUN apt-get -y install cmake
RUN git clone https://github.com/assimp/assimp.git
WORKDIR /usr/local/assimp/assimp
RUN cmake CMakeLists.txt
RUN make -j4
RUN cp /usr/local/assimp/assimp/lib/libassimp.so /usr/local/lib

#INSTALL MESHPARTY
RUN mkdir -p /usr/local/MeshParty
WORKDIR /usr/local/MeshParty
COPY . /usr/local/MeshParty
RUN pip install -r requirements.txt 
RUN pip install .

#OTHER PYTHON LIBRARIES 
RUN pip install pymeshfix --no-binary pymeshfix
RUN pip install pyassimp
RUN pip install matplotlib
RUN pip install Rtree
RUN pip install pandas
RUN pip install neuroglancer
RUN pip install annotationframeworkclient

#SPATIAL INDEX
RUN curl -L http://download.osgeo.org/libspatialindex/spatialindex-src-1.8.5.tar.gz | tar xz
WORKDIR /usr/local/MeshParty/spatialindex-src-1.8.5
RUN ./configure
RUN make
RUN make install
RUN ldconfig


#cgal
RUN apt-get install libgmp-dev -y
RUN apt-get install libmpfr-dev -y
WORKDIR /usr/local/cgal
RUN git clone https://github.com/CGAL/cgal.git
WORKDIR /usr/local/cgal/cgal
RUN rm -rf .git
RUN cmake .
RUN make

#need to replace this with forrest's scipy but still working on it
#RUN pip uninstall scipy
#RUN mkdir -p /usr/local/scipy
#WORKDIR /usr/local/scipy
#RUN git clone https://github.com/fcollman/scipy.git
#WORKDIR /usr/local/scipy/scipy
#RUN pip install .

WORKDIR /usr/local/MeshParty
