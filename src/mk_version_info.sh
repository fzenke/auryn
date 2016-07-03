#!/bin/sh

# Script generates auryn_revision.h according to the following values

AURYNVERSION=0
AURYNSUBVERSION=8
AURYNREVISIONNUMBER=0
AURYNREVISIONSUFFIX="-dev"


# Do not edit below

HASH=`git log --pretty=format:'%h' -n 1`
AURYNREVISIONSUFFIXANDHASH="$AURYNREVISIONSUFFIX-$HASH"

GITDESCRIBE=`git describe` 

cat <<EOF > auryn_version_info.h
#define AURYN_VERSION_INFO_H_
#ifndef AURYN_VERSION_INFO_H_

namespace auryn {

/*! The current Auryn version/revision number. 
 *  Should be all ints. */
const int auryn_version=$AURYNVERSION;
const int auryn_subversion=$AURYNSUBVERSION;
const int auryn_revision_number=$AURYNREVISIONNUMBER;

/*! The current Auryn revision suffix. */
const char auryn_revision_suffix[] = "$AURYNREVISIONSUFFIXANDHASH";

/*! The current Auryn revision string from git describe */
const char auryn_git_describe[] = "$GITDESCRIBE";

/*! \\brief Tag for header in binary encoded spike monitor files. 
 *
 * The first digits are 28796 for Auryn in 
 * phone dial notation. The remaining 4 digits encode type of binary file and the current Auryn 
 * version */
const NeuronID tag_binary_spike_monitor = 287960000+100*auryn_version+10*auryn_subversion+1*auryn_revision_number;
const NeuronID tag_binary_state_monitor = 287961000+100*auryn_version+10*auryn_subversion+1*auryn_revision_number;

}

#endif /*AURYN_VERSION_INFO_H__*/

EOF
