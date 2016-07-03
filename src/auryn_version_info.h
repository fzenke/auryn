#define AURYN_VERSION_INFO_H_
#ifndef AURYN_VERSION_INFO_H_

namespace auryn {

/*! The current Auryn version/revision number. 
 *  Should be all ints. */
const int auryn_version=0;
const int auryn_subversion=8;
const int auryn_revision_number=0;

/*! The current Auryn revision suffix. */
const char auryn_revision_suffix[] = "-dev-57cd2fb";

/*! The current Auryn revision string from git describe */
const char auryn_git_describe[] = "v0.8.0-dev-1-g57cd2fb";

/*! \brief Tag for header in binary encoded spike monitor files. 
 *
 * The first digits are 28796 for Auryn in 
 * phone dial notation. The remaining 4 digits encode type of binary file and the current Auryn 
 * version */
const NeuronID tag_binary_spike_monitor = 287960000+100*auryn_version+10*auryn_subversion+1*auryn_revision_number;
const NeuronID tag_binary_state_monitor = 287961000+100*auryn_version+10*auryn_subversion+1*auryn_revision_number;

}

#endif /*AURYN_VERSION_INFO_H__*/

