. load-modules.sh

for myport in {6819..11845}; do ! nc -z localhost ${myport} && break; done
echo "ssh -NL $myport:$(hostname):$myport $USER@login.rc.fas.harvard.edu"
jupyter-notebook --no-browser --port=$myport --ip='0.0.0.0'
