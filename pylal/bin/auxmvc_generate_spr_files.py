#!/usr/bin/python

# Copyright (C) 2011 Rahul Biswas, Ruslan Vaulin, Kari Hodge
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 2 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

#
# =============================================================================
#
#                                   Preamble
#
# =============================================================================
#

import os
import sys 
import string
from optparse import *
import glob
from commands import *
import subprocess
from pylal import auxmvc_utils
from pylal import git_version
import copy
import numpy

usage = """
        Tries to find the ratio
        """

def RoundRobin(glitch_list, clean_list, number):
    primary_glitch_set=glitch_list[number]
    primary_clean_set=clean_list[number]
    secondary_glitch_set_indices=[i for i in range(len(glitch_list)) if i != number]
    secondary_clean_set_indices=[i for i in range(len(clean_list)) if i != number]

     
    secondary_clean_set=clean_list[secondary_clean_set_indices[0]]
    for i in secondary_glitch_set_indices[1:]:
       secondary_clean_set=numpy.concatenate((secondary_clean_set,clean_list[i]))
    
    secondary_glitch_set=glitch_list[secondary_glitch_set_indices[0]]
    for i in secondary_glitch_set_indices[1:]:
       secondary_glitch_set=numpy.concatenate((secondary_glitch_set,glitch_list[i]))


    return  primary_clean_set, primary_glitch_set, secondary_clean_set, secondary_glitch_set

###########


def GenerateKWAuxGlitchTriggers(files):

   """Reads in the kw1-35_glitches_training sets and stores them in the memory 
   """
   KWAuxGlitchTriggers=auxmvc_utils.ReadKWAuxTriggers(files)
   return KWAuxGlitchTriggers


###########


def GenerateKWAuxCleanTriggers(files):

   """Reads in the kw1-35_signal_training sets and stores them in the memory 
   """
   KWAuxCleanTriggers=auxmvc_utils.ReadKWAuxTriggers(files)
   return KWAuxCleanTriggers


##########

def parse_command_line():

  """
  Parser function dedicated
  """
  parser = OptionParser(version=git_version.verbose_msg) 
  parser.add_option("-c","--clean-paramsfile", help="file with events of class zero")
  parser.add_option("-g","--glitch-paramsfile", help="file with events of class one")
  parser.add_option("-n","--roundrobin-number",default=10,type="int",help="number of round-robin training/testing sets to make")
  parser.add_option("","--dq-cats",action="store", type="string",default="ALL", help="Generate DQ veto categories" )
  parser.add_option("","--exclude-variables",action="store", type="string", default=None, help="Comma separated lits of variables that should be excluded from MVSC parameter list" )
  parser.add_option("","--output-tag",action="store",type="string", default=None, metavar=" OUTPUTTAG",\
      help="The output files will be named according to OUTPUTTAG" )

  (options,args) = parser.parse_args()

  return options, sys.argv[1:]


opts, args = parse_command_line()

###########

if not opts.clean_paramsfile:
  print >>sys.stderr, \
      "Must specify a clean triggers paramater text file"
  print >>sys.stderr, "Enter 'generate_mvsc_files.py --help' for usage"
  sys.exit(1)

if not opts.glitch_paramsfile:
  print >>sys.stderr, \
      "Must specify a glitch triggers paramater text file"
  print >>sys.stderr, "Enter 'generate_mvsc_files.py --help' for usage"
  sys.exit(1)



if opts.clean_paramsfile or opts.glitch_paramsfile is True:
  
  clean_paramsFile=[opts.clean_paramsfile]
  glitch_paramsFile=[opts.glitch_paramsfile]

  KWAuxCleanTriggers=GenerateKWAuxCleanTriggers(clean_paramsFile)
  KWAuxGlitchTriggers=GenerateKWAuxGlitchTriggers(glitch_paramsFile)



     
  dq_cats=opts.dq_cats.split(",")
  if opts.exclude_variables :
    exclude_variables_list = opts.exclude_variables.split(",")
  else:
    exclude_variables_list = None

  if opts.roundrobin_number:
    List_of_Clean_KW_Sets_cats = auxmvc_utils.split_array(KWAuxCleanTriggers, Nparts =int(opts.roundrobin_number))

  for cat in dq_cats:
 
    KW_Glitch_Triggers_cats=auxmvc_utils.getKWAuxTriggerFromDQCAT(KWAuxGlitchTriggers, cat)  
      
    if opts.roundrobin_number:

      List_of_Glitch_KW_Sets_cats = auxmvc_utils.split_array(KW_Glitch_Triggers_cats, Nparts = int(opts.roundrobin_number))
       
      for i in range(len(List_of_Glitch_KW_Sets_cats)):
    
        Primary_Clean_set_cats, Primary_Glitch_set_cats, Secondary_Clean_set_cats, Secondary_Glitch_set_cats=RoundRobin(List_of_Glitch_KW_Sets_cats, List_of_Clean_KW_Sets_cats,i)
        MVSC_evaluation_set_cats=auxmvc_utils.ConvertKWAuxToMVSC(KWAuxGlitchTriggers = Primary_Glitch_set_cats, KWAuxCleanTriggers = Primary_Clean_set_cats, ExcludeVariables = exclude_variables_list)
        MVSC_training_set_cats=auxmvc_utils.ConvertKWAuxToMVSC(KWAuxGlitchTriggers = Secondary_Glitch_set_cats, KWAuxCleanTriggers = Secondary_Clean_set_cats, ExcludeVariables = exclude_variables_list)
           
        output_evaluation=cat + "_" + opts.output_tag + "_set_" + str(i) + "_" + "evaluation.pat"
        auxmvc_utils.WriteMVSCTriggers(MVSC_evaluation_set_cats, output_filename = output_evaluation, Classified = False) 
                
        print output_evaluation

        output_training=cat + "_" + opts.output_tag + "_set_" + str(i) + "_" + "training.pat"
        auxmvc_utils.WriteMVSCTriggers(MVSC_training_set_cats, output_filename = output_training, Classified = False)











