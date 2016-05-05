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
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>

// Core simulator definitions
#include "auryn_global.h"
#include "auryn_definitions.h"
#include "AurynVector.h"
#include "System.h"
#include "SyncBuffer.h"
#include "Logger.h"
#include "SpikeDelay.h"

// Trace definitions
#include "LinearTrace.h"
#include "EulerTrace.h"

// Connection definitions
#include "Connection.h"
#include "SparseConnection.h"
#include "RateModulatedConnection.h"
#include "STDPConnection.h"
#include "STDPwdConnection.h"
#include "SymmetricSTDPConnection.h"
#include "STPConnection.h"
#include "ABSConnection.h"
#include "DuplexConnection.h"
#include "TripletConnection.h"
#include "TripletDecayConnection.h"
#include "TripletScalingConnection.h"
#include "IdentityConnection.h"

// Spiking and Neuron group definitions
#include "AIF2Group.h"
#include "IafPscDeltaGroup.h"
#include "IFGroup.h"
#include "AIFGroup.h"
#include "AdExGroup.h"
#include "CubaIFGroup.h"
#include "TIFGroup.h"
#include "SIFGroup.h"
#include "SpikingGroup.h"
#include "NeuronGroup.h"
#include "PoissonGroup.h"
#include "ProfilePoissonGroup.h"
#include "StructuredPoissonGroup.h"
#include "CorrelatedPoissonGroup.h"
#include "MovingBumpGroup.h"
#include "FileModulatedPoissonGroup.h"
#include "AuditoryBeepGroup.h"
#include "StimulusGroup.h"
#include "FileInputGroup.h"


// Checker definitions
#include "Checker.h"
#include "RateChecker.h"
#include "WeightChecker.h"

// Monitor and stimulator definitions
#include "Monitor.h"
#include "GabaMonitor.h"
#include "VoltageMonitor.h"
#include "SpikeMonitor.h"
#include "BinarySpikeMonitor.h"
#include "DelayedSpikeMonitor.h"
#include "RealTimeMonitor.h"
#include "RateMonitor.h"
#include "PopulationRateMonitor.h"
#include "StateMonitor.h"
#include "WeightSumMonitor.h"
#include "PatternMonitor.h"
#include "WeightPatternMonitor.h"
#include "WeightStatsMonitor.h"
#include "WeightMonitor.h"
#include "WeightMatrixMonitor.h"
#include "PoissonStimulator.h"
#include "NormalStimulator.h"
#include "PatternStimulator.h"
#include "CurrentInjector.h"


#endif /*AURYN_H__*/
