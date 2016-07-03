#define AURYN_VERSION_INFO_H_
#ifndef AURYN_VERSION_INFO_H_

#include <string>
#include "auryn_definitions.h"

namespace auryn {

extern static const int auryn_version;
extern static const int auryn_subversion;
extern static const int auryn_revision_number;
extern const char auryn_revision_suffix[];
extern const char auryn_git_describe[];
extern static const NeuronID tag_binary_spike_monitor;
extern static const NeuronID tag_binary_state_monitor;

}

#endif /*AURYN_VERSION_INFO_H__*/

