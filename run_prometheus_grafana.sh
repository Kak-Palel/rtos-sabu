#!/bin/bash

workspace_dir=$(pwd)

sudo systemctl start grafana-server
# python3 -m flask --app app run 
~/prometheus-3.5.0.linux-amd64/prometheus --config.file="$workspace_dir/prometheus.yml"