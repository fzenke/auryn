#!/bin/sh

# Script generates auryn_revision.cpp according to the following values

AURYNVERSION=0
AURYNSUBVERSION=8
AURYNREVISIONNUMBER=1
AURYNREVISIONSUFFIX="dev"


# Do not edit below

OUTDIR=`dirname $0`

HASH=`git log --pretty=format:'%h' -n 1`
AURYNREVISIONSUFFIXANDHASH="$AURYNREVISIONSUFFIX-$HASH"

GITDESCRIBE=`git describe` 

cat <<EOF > $OUTDIR/AurynVersion.cpp
// Please note that his file is created by mk_version_info.sh 
// Changes to this file may thous be overwritten
#include "AurynVersion.h"

namespace auryn {

    int AurynVersion::version = $AURYNVERSION;
    int AurynVersion::subversion = $AURYNSUBVERSION;
    int AurynVersion::revision_number = $AURYNREVISIONNUMBER;
    NeuronID AurynVersion::tag_binary_spike_monitor = 287960000+100*$AURYNVERSION+10*$AURYNSUBVERSION+1*$AURYNREVISIONNUMBER; //!< file signature for BinarySpikeMonitor files
    AurynState AurynVersion::tag_binary_state_monitor = 61000+100*$AURYNVERSION+10*$AURYNSUBVERSION+1*$AURYNREVISIONNUMBER; //!< file signature for BinaryStateMonitor files
    std::string AurynVersion::revision_suffix = "$AURYNREVISIONSUFFIXANDHASH";
    std::string AurynVersion::git_describe = "$GITDESCRIBE";

}

EOF


