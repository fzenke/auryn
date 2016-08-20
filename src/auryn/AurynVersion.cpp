// Please note that his file is created by mk_version_info.sh 
// Changes to this file may thous be overwritten
#include "AurynVersion.h"

namespace auryn {

    int AurynVersion::version = 0;
    int AurynVersion::subversion = 8;
    int AurynVersion::revision_number = 0;
    NeuronID AurynVersion::tag_binary_spike_monitor = 287960000+100*0+10*8+1*0; //!< file signature for BinarySpikeMonitor files
    AurynState AurynVersion::tag_binary_state_monitor = 61000+100*0+10*8+1*0; //!< file signature for BinaryStateMonitor files
    std::string AurynVersion::revision_suffix = "-beta-5e0503d";
    std::string AurynVersion::git_describe = "v0.8.0-dev-262-g5e0503d";

}

