//------------------------------------------------------------------------
// File and Version Information:
//
// Description:
//      Class MySqliteReader :
//         Read in data info from a file and compose SprData.
//         Rules:
//         -----
//         Text to the right of '#' will be treated as comments.
//         The required format of an input file in mode 1 is:
//
//            # comment
//            unsigned
//            name1 name2 name3
//            double double double int
//            double double double int
//            double double double int
//            ...
//
//         where the first unsigned is the number of dimensions,
//         the names are the variable names,
//         the doubles on each line are coordinates of one point
//         with the number of doubles being equal to the number of
//         dimensions, and the last int on each line is the 
//         category of this point.
//
//         In mode 2 the input file must look like:
//            # comment
//            unsigned
//            name1 name2 name3
//            double double double
//            int
//            double double double
//            int
//            double double double
//            int
//
//         In mode 3 the input file must look like:
//            # comment
//            unsigned
//            name1 name2 name3
//            double double double charge
//            class
//            double double double charge
//            class
//            double double double charge
//            class
//	   
//         ...and the input class identifier is determined by
//	   multiplying charge*class. This is done for tagger compatibility.
//
//         Mode 4 is identical to mode 1, except that for each sample,
//         the class is preceded by a weight (as the second-to-last
//         number on each line).  For example:
//
//            unsigned
//            name1 name2 name3
//            double double double double int
//            double double double double int
//            double double double double int
//            ...
//
//         Mode 5 is identical to mode 1, and mode 6 is respectively 
//         identical to mode 4, except variable names appear on separate 
//         lines with each variable name taking one line.
//
//            unsigned
//            name1 
//            name2 
//            name3
//            double double double int
//            double double double int
//            double double double int
//
//
// Environment:
//      Software developed for the BaBar Detector at the SLAC B-Factory.
//
// Author List:
//      Ilya Narsky                     Original author
//
// Copyright Information:
//      Copyright (C) 2005              California Institute of Technology
//
//------------------------------------------------------------------------
 
#ifndef _MySqliteReader_HH
#define _MySqliteReader_HH

#include "StatPatternRecognition/SprAbsReader.hh"

#include <set>
#include <vector>
#include <string>

class SprAbsFilter;
class SprPreFilter;


class MySqliteReader : public SprAbsReader
{
public:
  virtual ~MySqliteReader() {}

  explicit MySqliteReader(const std::string& sqliteQuery, SprPreFilter* filter = 0);

  bool chooseVars(const std::set<std::string>& vars) { 
    include_ = vars;
    return true;
  }

  void chooseAll() { include_.clear(); exclude_.clear(); }

  bool chooseAllBut(const std::set<std::string>& vars) { 
    exclude_ = vars;
    return true;
  }

  SprAbsFilter* read(const char* filename);

private:
  std::set<std::string> include_;
  std::set<std::string> exclude_;
  std::string sqliteQuery_;
};

#endif

