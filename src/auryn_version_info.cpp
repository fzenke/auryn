#include "auryn_version_info.h"
#include "auryn_definitions.h"

namespace auryn {

/*! The current Auryn version/revision number. */
static const int auryn_version=0;
static const int auryn_subversion=8;
static const int auryn_revision_number=0;

/*! The current Auryn revision suffix. */
const char auryn_revision_suffix[] = "-dev-eab7157";

/*! The current Auryn revision string from git describe */
const char auryn_git_describe[] = "v0.8.0-dev-1-geab7157";

/*! \brief Tag for header in binary encoded spike monitor files. 
 *
 * The first digits are 28796 for Auryn in 
 * phone dial notation. The remaining 4 digits encode type of binary file and the current Auryn 
 * version */
static const NeuronID tag_binary_spike_monitor = 287960000+100*auryn_version+10*auryn_subversion+1*auryn_revision_number;
static const NeuronID tag_binary_state_monitor = 287961000+100*auryn_version+10*auryn_subversion+1*auryn_revision_number;
}

