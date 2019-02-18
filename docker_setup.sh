docker build -t sharmi/meshparty .
docker kill sharmi_meshparty
docker rm sharmi_meshparty
docker run -d --name sharmi_meshparty \
-v /home/sharmishtaas/neural_coding/projects/code/MeshParty_branches/cgal_meshparty/MeshParty:/usr/local/MeshParty \
-v /etc/hosts:/etc/hosts \
-p 9777:9777 \
-e "PASSWORD=$JUPYTERPASSWORD" \
-i -t sharmi/meshparty \
/bin/bash -c "jupyter notebook --allow-root "
#--notebook-dir=/home/sharmishtaas/neural_coding/projects/code/MeshParty_branches/cgal_meshparty/MeshParty"
