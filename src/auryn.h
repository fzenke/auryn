/* 
* Copyright 2014 Friedemann Zenke
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

#include "auryn_global.h"
#include "auryn_definitions.h"
#include "System.h"
#include "Logger.h"
#include "SpikingGroup.h"
#include "NeuronGroup.h"
#include "IFGroup.h"
#include "AIFGroup.h"
#include "PoissonGroup.h"
#include "CorrelatedPoissonGroup.h"
#include "SparseConnection.h"
#include "STPConnection.h"
#include "IdentityConnection.h"
#include "TripletConnection.h"
#include "TripletDecayConnection.h"
#include "WeightMonitor.h"
#include "VoltageMonitor.h"
#include "WeightMatrixMonitor.h"
#include "PopulationRateMonitor.h"
#include "SpikeMonitor.h"
#include "RealTimeMonitor.h"
#include "RateChecker.h"
#include "FileInputGroup.h"
#include "PatternMonitor.h"
#include "PatternStimulator.h"


#endif /*AURYN_H__*/
