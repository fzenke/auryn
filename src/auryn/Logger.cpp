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

#include "Logger.h"
#include "auryn_definitions.h"

using namespace auryn;

Logger::Logger(std::string filename, int rank, LogMessageType console, LogMessageType file)
{
	console_out = console; 
	file_out  = file;
	set_rank(rank);
	fname = filename;

	outfile.open(fname.c_str(),std::ios::out);
	if (!outfile) {
		std::cerr << "Can't open output logger output file " << fname << std::endl;
	    throw AurynOpenFileException();
	}


	std::stringstream oss;
	oss << "Logger started on Rank " << local_rank;
	msg(oss.str(),NOTIFICATION);
}

Logger::~Logger()
{
	std::stringstream oss;
	oss << "Logger stopped";
	msg(oss.str(),NOTIFICATION);
	outfile.close();
}

void Logger::set_console_loglevel(LogMessageType level)
{
	console_out = level;
}

void Logger::set_logfile_loglevel(LogMessageType level)
{
	file_out = level;
}

void Logger::set_debugging_mode()
{
	set_console_loglevel(EVERYTHING);
	set_logfile_loglevel(EVERYTHING);
}

void Logger::msg( std::string text, LogMessageType type, bool global, int line, std::string srcfile )
{
	time_t rawtime;
	struct tm * timeinfo;
	time ( &rawtime );
	timeinfo = localtime ( &rawtime );
	char tbuffer [80];
	strftime (tbuffer,80,"%x %X",timeinfo);

	std::stringstream oss;
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
			if ( type >= AURYN_LOGGER_CERRLEVEL )
				if ( global ) {
					std::cerr << "(!!) " << text << std::endl;
				} else {
					std::cerr << "(!!) " << "on rank " << local_rank << ": " << text << std::endl;
				}
			else
				std::cout << "(" << std::setw(2) << local_rank << ") " << text << std::endl;
		} 
	}

	if ( type >= file_out)
		outfile << oss.str() << std::endl;

	last_message = text;
}

void Logger::progress( std::string text )
{
	msg(text, PROGRESS, true);
}

void Logger::info( std::string text )
{
	msg(text, INFO);
}

void Logger::notification( std::string text )
{
	msg(text, NOTIFICATION);
}

void Logger::warning( std::string text )
{
	msg(text, WARNING);
}

void Logger::error( std::string text )
{
	msg(text, ERROR);
}

void Logger::verbose( std::string text, bool global, int line, std::string srcfile )
{
	debug(text, global, line, srcfile );
}

void Logger::debug( std::string text, bool global, int line, std::string srcfile )
{
	msg(text, VERBOSE, global, line, srcfile );
}

void Logger::set_rank(int rank)
{
	local_rank = rank;
}
