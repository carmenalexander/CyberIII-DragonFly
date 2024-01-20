#!/bin/bash

set -e

# Get the full path of the binary
ARCHIVE=$(realpath "$1")
VERSION="$2"
echo "Preparing $ARCHIVE"

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Setup RPM build environment in a unique subdirectory under /tmp
RPM_ROOT=$(mktemp -d /tmp/rpmbuild_XXXXXX)
echo "Working dir is $RPM_ROOT"
mkdir -p $RPM_ROOT/{BUILD,RPMS,SOURCES,SPECS}

# Put the archive and configuration files to the SOURCES directory
ln -s "$ARCHIVE" -t "$RPM_ROOT/SOURCES/"
cp $SCRIPT_DIR/dragonfly.service $RPM_ROOT/SOURCES/
cp $SCRIPT_DIR/dragonfly.conf $RPM_ROOT/SOURCES/

# decode pkg name from archive name i.e /dragonfly-db-x86_64.tar.gz or /dragonfly-x86_64.tar.gz
PKG_NAME=$(echo "$ARCHIVE" | sed -e 's/.*\///' -e 's/-.*//')
if [[ $ARCHIVE == *"-dbg"* ]]; then
    PKG_NAME="${PKG_NAME}-dbg"
fi

cp $SCRIPT_DIR/dragonfly.spec $RPM_ROOT/SPECS/

rpmbuild --define "_topdir $RPM_ROOT" --define "pkg_name $PKG_NAME" --define "version $VERSION" -bb "$RPM_ROOT/SPECS/dragonfly.spec"
mv $RPM_ROOT/RPMS/*.rpm ./
