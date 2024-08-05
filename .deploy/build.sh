#!/usr/bin/env bash

BASEDIR = "$(dirname "$(readlink -f "$0")")"
if [ "$1" == "--rebuild" ]; then rm -rf "$BASEDIR/build"; fi
mkdir -p $BASEDIR/build

cp $BASEDIR/docker-compose.yml $BASEDIR/build

docker build -t bleyn/2d_segmentation:latest $BASEDIR/..
if [ ! -e "$BASEDIR/build/2d_segmentation.tar.gz" ]; then
	docker image save bleyn/2d_segmentation:latest | gzip > $BASEDIR/build/2d_segmentation.tar.gz
fi

mkdir -p $BASEDIR/build/data
cp -r $BASEDIR/../data $BASEDIR/build/data/data

echo "!/usr/bin/env bash
BASEDIR = \"$(dirname \"\$(readlink -f \"\$0\")\")\"

docker load < \$BASEDIR/2d_segmentation.tar.gz
docker-compose -f \$BASEDIR/build/docker-compose.yml up -d" > $BASEDIR/build/deploy.sh