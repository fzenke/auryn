/* 
* Copyright 2014-2016 Friedemann Zenke
*
* This file is part of Auryn, a simulation package for plastic
* spiking neural networks.
* 
* Auryn is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
* 
* Auryn is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
* 
* You should have received a copy of the GNU General Public License
* along with Auryn.  If not, see <http://www.gnu.org/licenses/>.
*
* If you are using Auryn or parts of it for your work please cite:
* Zenke, F. and Gerstner, W., 2014. Limits to high-speed simulations 
* of spiking neural networks using general-purpose computers. 
* Front Neuroinform 8, 76. doi: 10.3389/fninf.2014.00076
*/

#ifndef AURYN_H_
#define AURYN_H_

#include <iostream>
#include <iomanip>
#include <stdlib.h>
#include <string>

#include <boost/program_options.hpp>

#ifdef AURYN_CODE_USE_MPI
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#endif // AURYN_CODE_USE_MPI

// Core simulator definitions
#include "auryn/auryn_global.h"
#include "auryn/auryn_definitions.h"
#include "auryn/AurynVector.h"
#include "auryn/AurynDelayVector.h"
#include "auryn/System.h"
#include "auryn/Logger.h"
#include "auryn/SpikeDelay.h"
#ifdef AURYN_CODE_USE_MPI
#include "auryn/SyncBuffer.h"
#endif // AURYN_CODE_USE_MPI


// Trace definitions
#include "auryn/LinearTrace.h"
#include "auryn/EulerTrace.h"

// Connection definitions
#include "auryn/Connection.h"
#include "auryn/SparseConnection.h"
#include "auryn/RateModulatedConnection.h"
#include "auryn/STDPConnection.h"
#include "auryn/STDPwdConnection.h"
#include "auryn/SymmetricSTDPConnection.h"
#include "auryn/STPConnection.h"
#include "auryn/ABSConnection.h"
#include "auryn/DuplexConnection.h"
#include "auryn/TripletConnection.h"
#include "auryn/LPTripletConnection.h"
#include "auryn/TripletDecayConnection.h"
#include "auryn/TripletScalingConnection.h"
#include "auryn/IdentityConnection.h"
#include "auryn/AllToAllConnection.h"

// Spiking and input group definitions
#include "auryn/SpikingGroup.h"
#include "auryn/NeuronGroup.h"
#include "auryn/PoissonGroup.h"
#include "auryn/PoissonSpikeInjector.h"
#include "auryn/FileInputGroup.h"
#include "auryn/FileModulatedPoissonGroup.h"
#include "auryn/StimulusGroup.h"
#include "auryn/SpikeTimingStimGroup.h"
#include "auryn/ProfilePoissonGroup.h"
#include "auryn/StructuredPoissonGroup.h"
#include "auryn/CorrelatedPoissonGroup.h"
#include "auryn/MovingBumpGroup.h"
#include "auryn/AuditoryBeepGroup.h"

// NeuronGroups
#include "auryn/IFGroup.h"
#include "auryn/CubaIFGroup.h"
#include "auryn/TIFGroup.h"
#include "auryn/AIFGroup.h"
#include "auryn/AIF2Group.h"
#include "auryn/AdExGroup.h"
#include "auryn/IafPscDeltaGroup.h"
#include "auryn/IafPscExpGroup.h"
#include "auryn/IzhikevichGroup.h"


// Checker definitions
#include "auryn/Checker.h"
#include "auryn/RateChecker.h"
#include "auryn/WeightChecker.h"

// Monitor and stimulator definitions
#include "auryn/Device.h"
#include "auryn/Monitor.h"
#include "auryn/VoltageMonitor.h"
#include "auryn/SpikeMonitor.h"
#include "auryn/BinarySpikeMonitor.h"
#include "auryn/BinaryStateMonitor.h"
#include "auryn/DelayedSpikeMonitor.h"
#include "auryn/RealTimeMonitor.h"
#include "auryn/RateMonitor.h"
#include "auryn/PopulationRateMonitor.h"
#include "auryn/StateMonitor.h"
#include "auryn/WeightSumMonitor.h"
#include "auryn/PatternMonitor.h"
#include "auryn/WeightPatternMonitor.h"
#include "auryn/WeightStatsMonitor.h"
#include "auryn/WeightMonitor.h"
#include "auryn/WeightMatrixMonitor.h"
#include "auryn/PoissonStimulator.h"
#include "auryn/NormalStimulator.h"
#include "auryn/PatternStimulator.h"
#include "auryn/CurrentInjector.h"


#endif /*AURYN_H__*/
