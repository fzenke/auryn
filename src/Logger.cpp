/* 
* Copyright 2014-2015 Friedemann Zenke
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

#include "Logger.h"
#include "auryn_definitions.h"

Logger::Logger(string filename, int rank, LogMessageType console, LogMessageType file)
{
	console_out = console; 
	file_out  = file;
	set_rank(rank);
	fname = filename;

	outfile.open(fname.c_str(),ios::out);
	if (!outfile) {
		std::cerr << "Can't open output logger output file " << fname << std::endl;
	    throw AurynOpenFileException();
	}


	stringstream oss;
	oss << "Logger started on Rank " << local_rank;
	msg(oss.str(),NOTIFICATION);
}

Logger::~Logger()
{
	stringstream oss;
	oss << "Logger stopped";
	msg(oss.str(),NOTIFICATION);
	outfile.close();
}

void Logger::msg( string text, LogMessageType type, bool global, int line, string srcfile )
{
	time_t rawtime;
	struct tm * timeinfo;
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	char tbuffer [80];
	strftime (tbuffer,80,"%x %X",timeinfo);

	stringstream oss;
	oss << tbuffer << ":: ";
	switch ( type ) {
		case WARNING:
			oss << "!! WARNING: ";
			break;
		case ERROR:
			oss << "!! ERROR: ";
			break;
		default:
			break;
	}

	oss << text; // actual text output

	if ( line >= 0 ) {
		oss << " @ line " << line << " in " << srcfile;
	}

	if ( type >= console_out && ( !global || (global && local_rank == 0) ) ) {
		if ( last_message != text ) {
			if ( type >= CERRLEVEL )
				if ( global ) {
					cerr << "(!!) " << text << endl;
				} else {
					cerr << "(!!) " << "on rank " << local_rank << ": " << text << endl;
				}
			else
				cout << "(" << setw(2) << local_rank << ") " << text << endl;
		} 
	}

	if ( type >= file_out)
		outfile << oss.str() << endl;

	last_message = text;
}

void Logger::parameter(string name, double value) 
{
	stringstream oss;
	oss << scientific << "  Parameter " << name << "=" << value;
	msg(oss.str(),SETTINGS,true);
}

void Logger::parameter(string name, int value) 
{
	stringstream oss;
	oss << scientific << "Setting " << name << "=" << value;
	msg(oss.str(),SETTINGS,true);
}

void Logger::parameter(string name, string value) 
{
	stringstream oss;
	oss.precision(9);
	oss << scientific << "Setting " << name << "=" << value;
	msg(oss.str(),SETTINGS,true);
}

void Logger::set_rank(int rank)
{
	local_rank = rank;
}
