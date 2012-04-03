import os
import sys
import numpy
import copy
import pickle

def ROC(clean_ranks, glitch_ranks):
  """
  Calculates ROC curves based on the ranks assigned by a classifier to clean and glitch aux triggers.
  """
  clean_ranks_sorted = numpy.sort(clean_ranks)
  glitch_ranks_sorted = numpy.sort(glitch_ranks)
  number_of_false_alarms=[]
  number_of_true_alarms=[]
  FAP = []
  Eff = []
  for i,rank in enumerate(clean_ranks_sorted):
    FAP.append(compute_FAP(clean_ranks_sorted, glitch_ranks_sorted, rank))
    Eff.append(compute_Eff(clean_ranks_sorted, glitch_ranks_sorted, rank))

  return numpy.asarray(FAP), numpy.asarray(Eff)


def compute_FAP(clean_ranks, glitch_ranks, rank):
  """
  Compute false alarm probability for a given rank using array of ranks for clean and glitch samples.
  Input arrays clean_ranks and glitch_ranks mst be sorted in ascending order.
  """
  number_of_false_alarms = len(clean_ranks) - numpy.searchsorted(clean_ranks,rank)  
  FAP = number_of_false_alarms / float(len(clean_ranks))
  return FAP
  
def compute_Eff(clean_ranks, glitch_ranks, rank):
  """
  Compute detection probability for a given rank using array of ranks for clean and glitch samples.
  Input arrays clean_ranks and glitch_ranks mst be sorted in ascending order.
  """  
  number_of_true_alarms = len(glitch_ranks) - numpy.searchsorted(glitch_ranks,rank)
  Eff = number_of_true_alarms / float(len(glitch_ranks))
  return Eff
  

def split_array(array, Nparts = 2):
  """ 
  Splits 2-d record array in N equal parts. 
  If the the number of elements in array is not divisible by Nparts, all but last sub arrays are equal.
  It returns list of sub-arrays.
  """
  
  subarrays = []
  n = int(len(array) / float(Nparts))
  
  for i in range(Nparts):
    if i == Nparts - 1:
      subarrays.append(array[:][i*n:])
    else:
      subarrays.append(array[:][i*n:(i+1)*n])
  
  return subarrays
 
def get_clean_samples(Triggers):
  """
  Returns only clean samples from the set of random samples. By definition, a sample in unclean if there is a KW trigger with 0.1 seconds time window. 
  """ 
  
  return Triggers[numpy.nonzero(Triggers['unclean'] == 0.0)[0],:]

def getKWAuxTriggerFromDQCAT(Triggers, DQ_category):

    if DQ_category == 'DQ2':
      return Triggers[numpy.nonzero(Triggers['DQ2'] == 1.0)[0],:]
    elif DQ_category == 'DQ3':
      return Triggers[numpy.nonzero((Triggers['DQ2'] == 0.0) * (Triggers['DQ3'] == 1.0))[0],:]  
    elif DQ_category == 'DQ4':
      return Triggers[numpy.nonzero((Triggers['DQ2'] == 0.0) * (Triggers['DQ3'] == 0.0) * (Triggers['DQ4'] == 1.0))[0],:]
    elif DQ_category == 'DQ23':
      return Triggers[numpy.nonzero((Triggers['DQ2'] == 1.0) + (Triggers['DQ3'] == 1.0))[0],:] 
    elif DQ_category == 'DQ234':
      return Triggers[numpy.nonzero((Triggers['DQ2'] == 1.0) + (Triggers['DQ3'] == 1.0) + (Triggers['DQ4'] == 1.0))[0],:]
    elif DQ_category == 'aDQ2':
      return Triggers[numpy.nonzero(Triggers['DQ2'] == 0.0)[0],:]
    elif DQ_category == 'aDQ23':
      return Triggers[numpy.nonzero((Triggers['DQ2'] == 0.0) * (Triggers['DQ3'] == 0.0))[0],:]
    elif DQ_category == 'aDQ234':
      return Triggers[numpy.nonzero((Triggers['DQ2'] == 0.0) * (Triggers['DQ3'] == 0.0) * (Triggers['DQ4'] == 0.0))[0],:]
    elif DQ_category == 'ALL':
      return Triggers
    else:
      raise ValueError("Unknown DQ category") 

def ReadKWAuxTriggers(files):
  
  """
  Reads in KW auxiliary triggers from files. Triggers are storead in the 2-D array.
  The rows of the array are labelled by the names of the variables, which are read off of the first line of the input file.
  The columns are populated by the values of the corresponding variables. 
  Every line (except the first) of the input file(s) corresponds to a column (or a KW trigger) in the array. 
  """
  for (i,f) in enumerate(files):
    flines = open(f).readlines()
    variables = flines[0].split()
    formats = ['g8' for a in range(len(variables))]
    if i > 0:
      KWAuxTriggers  = numpy.concatenate((KWAuxTriggers ,numpy.loadtxt(f,skiprows=1, dtype={'names': variables,'formats':formats})),axis=0)
    else:
      KWAuxTriggers  = numpy.loadtxt(f,skiprows=1, dtype={'names': variables,'formats':formats})
        
  return KWAuxTriggers  
 
def ShuffleKWAuxTriggers(KWAuxTriggers, dT=60.0):
  """
  Shuffle segmented trigger packets. The trigger packets are segmented with each dT seconds usig G
PS time.
  """
  gps = KWAuxTriggers['GPS']
  gps_start_time = int(numpy.min(gps))
  gps_end_time = int(numpy.max(gps))+1
  duration = (gps_end_time - gps_start_time)
  n_minutes = int(duration / dT) + 1
  start_times = gps_start_time + dT * numpy.arange(n_minutes)
  numpy.random.shuffle(start_times)
  end_times = start_times + dT

  start_indexes = numpy.searchsorted(gps, start_times)
  end_indexes = numpy.searchsorted(gps, end_times)
  n_triggers = len(KWAuxTriggers)
  ShuffledKWAuxTriggers = numpy.empty((n_triggers,), dtype=KWAuxTriggers.dtype)
  current_start_index = 0
  current_end_index = 0
  for (start_index, end_index) in zip(start_indexes, end_indexes):
    if start_index < end_index:
      current_end_index = current_start_index + end_index - start_index
      ShuffledKWAuxTriggers[:][current_start_index:current_end_index] = KWAuxTriggers[:][start_index:end_index]
      current_start_index = current_end_index 

  return ShuffledKWAuxTriggers
  
def FilterKWAuxTriggers(KWAuxTriggers,excludeparameters,excludechannels):

  """
  Irrelevant channels and trigger parameters are excluded.
  excludechannels is a list of irrelevant channel names to be excluded.
  excludeparameters is a list of comma separated parameters. i.e. dur,freq,npts
  """

  KWAuxvariables = list(KWAuxTriggers.dtype.names)
  filteredvariables = copy.copy(KWAuxvariables)

  # both lists are empty, return original triggers
  if not excludeparameters and  excludechannels:
    return KWAuxTriggers
 
  if excludeparameters:
    ExParams = excludeparameters.split(",")
    for exp in ExParams:
      for pvar in KWAuxvariables:
        if exp.strip() in pvar.strip():
          if pvar in filteredvariables:
            filteredvariables.remove(pvar)

  if excludechannels:
    for exc in excludechannels:
      for cvar in KWAuxvariables:
        if cvar in filteredvariables:
          if exc.strip() in cvar.strip():
            filteredvariables.remove(cvar)

  #print "the list of included variable after filtering"
  #for (i,f) in enumerate(filteredvariables):
    #print "%i-th variable to be included : %s" % (i+1,f)

  n_triggers = len(KWAuxTriggers)
  formats = ['g8' for a in range(len(filteredvariables))]
  FilteredKWAuxTriggers = numpy.empty((n_triggers,),dtype={'names':filteredvariables,'formats':formats})

  for fvariable in filteredvariables:
    FilteredKWAuxTriggers[fvariable] = KWAuxTriggers[fvariable]

  return FilteredKWAuxTriggers




def ConvertKWAuxToMVSC(KWAuxGlitchTriggers, KWAuxCleanTriggers, ExcludeVariables = None):

  """
  Converts KW auxiliary triggers into MVSC triggers.
  KWAuxGlitchTriggers - KW triggers corresponding to glitches in DARM
  KWAuxCleanTriggers - KW triggers correspondingto clean DARM data. 
  """
  if ExcludeVariables:
    KWvariables = list(KWAuxGlitchTriggers.dtype.names)
    for variable in ExcludeVariables:
      KWvariables.remove(variable)

  # remove  GPS time and add GPS seconds and miliseconds
  if 'GPS' in KWvariables: 
    KWvariables.remove('GPS')
  
  MVSCvariables = ['index', 'i', 'w', 'GPS_s', 'GPS_ms']+ KWvariables + ['glitch-rank']
  formats = ['i','i','g8','i', 'i'] + ['g8' for a in range(len(MVSCvariables) - 5)]
  n_triggers = len(KWAuxGlitchTriggers) + len(KWAuxCleanTriggers)
  
  i_row = numpy.concatenate((numpy.ones(len(KWAuxGlitchTriggers)), numpy.zeros(len(KWAuxCleanTriggers))))
  index_row = numpy.arange(1, n_triggers + 1)
  w_row = numpy.ones(n_triggers)
  glitch_rank_row = numpy.zeros(n_triggers)
  
  MVSCTriggers = numpy.empty((n_triggers,), dtype={'names': MVSCvariables,'formats':formats})
  MVSCTriggers['index'] = index_row
  MVSCTriggers['i'] = i_row
  MVSCTriggers['w'] = w_row
  MVSCTriggers['glitch-rank'] = glitch_rank_row
  #set seconds and nanoseconds columns
  MVSCTriggers['GPS_s'] = (numpy.concatenate((KWAuxGlitchTriggers['GPS'], KWAuxCleanTriggers['GPS'])) // 1.0).astype('int')
  MVSCTriggers['GPS_ms'] = ((numpy.around((numpy.concatenate((KWAuxGlitchTriggers['GPS'], KWAuxCleanTriggers['GPS'])) % 1.0), decimals=3) * 10**3) // 1.0).astype('int')
  for variable in MVSCvariables:
    if not variable in ['index', 'i', 'w', 'glitch-rank', 'GPS_s', 'GPS_ms']:
      MVSCTriggers[variable] = numpy.concatenate((KWAuxGlitchTriggers[variable], KWAuxCleanTriggers[variable]))
    

  return MVSCTriggers
  
def WriteMVSCTriggers(MVSCTriggers, output_filename, Classified = False):
  
  """
  Write MVSC triggers to file.
  If Classified = False, triggers are treated as unclassfied and saved in the input file for MVSC.
  If Classified = True, triggers as saved in the same format as output of MVSC.   
  """   
  n_triggers = len(MVSCTriggers)
  if not Classified:
    Unclassified_variables = list(MVSCTriggers.dtype.names)
    for var in ['index', 'i', 'w', 'glitch-rank']:
      Unclassified_variables.remove(var)
    Unclassified_variables.append('i')
    formats = []
    for var in Unclassified_variables:
      if var in ['GPS_s', 'GPS_ms', 'i']:
        formats.append('i')
      else:
        formats.append('g8')
      
    #formats = ['g8' for a in range(len(Unclassified_variables) - 1)] + ['i']
    Triggers = numpy.empty((n_triggers,), dtype={'names': Unclassified_variables,'formats':formats})
    
    for variable in Unclassified_variables:
      Triggers[variable] = MVSCTriggers[variable]

  else:
    Triggers = MVSCTriggers
   
  
  file = open(output_filename, "w")
  
  if Classified:
    first_line = " ".join(list(Triggers.dtype.names))
    file.write(first_line + "\n")
  else:
    first_line = str(len(list(Triggers.dtype.names)[:-1]))
    second_line = " ".join(list(Triggers.dtype.names)[:-1])
    file.write(first_line + "\n")
    file.write(second_line + "\n")
  
  for i in range(n_triggers):
    line = " ".join([str(var) for var in Triggers[i]])
    file.write(line + "\n")

  file.close()    
  
  
def ReadMVSCTriggers(files):

  """
  Reads in MVSC triggers from files. MVSC triggers are storead in the 2-D array.
  The rows of the array are labelled by the names of the variables, which are read off of the first line of the input file.
  The columns are populated by the values of the corresponding variables. 
  Every line (except the first) of the input file(s) corresponds to a column (or a MVSC trigger) in the array. 
  """
  for (i,f) in enumerate(files):
    flines = open(f).readlines()
    variables = flines[0].split()
    formats = []
    for var in variables:
      if var in ['GPS_s', 'GPS_ms', 'i', 'index']:
        formats.append('i')
      else:
        formats.append('g8')
    #formats = ['i','i']+['g8' for a in range(len(variables)-2)]
    if i > 0:
      MVSCTriggers  = numpy.concatenate((MVSCTriggers ,numpy.loadtxt(f,skiprows=1, dtype={'names': variables,'formats':formats})),axis=0)
    else:
      MVSCTriggers  = numpy.loadtxt(f,skiprows=1, dtype={'names': variables,'formats':formats})
        
  return MVSCTriggers  


def LoadCV(filename):

  """
  Reads in the pickled output of CV data (eg: kwl1-35.track.9.pickle) and inverts the data storage. Returns a list of gwtrg's removed by the Cveto method.
  returned gwtrg's are labeled by tcent (and only tcent) and are associated with a given vconfig (vchan, vthr, vwin) and vstats (dsec, c_dsec, etc)
  input arg:
    filename: the file that is to be loaded. This must be a string
  output arg:
    gwtrg_vtd_tcent: a list of gwtrgs (labeled by tcent) with associated vconfig and vstats. the storage structure is: under each column: [tcent, [vconfig], [vstats]]
      check below for exact formats of vconfig and vstats. 
  """  

  # define column names for KW trig's
  col_kw   = {'tstart':0, 'tstop':1, 'tcent':2, 'fcent':3, 'uenergy':4, 'nenergy':5, 'npix':6, 'signif':7}

  # load and unpickle the .track.run.pickle file to be loaded
  file = open(filename, 'r')
  tfp = pickle.load(file)

  # define a dictionary that labels the indexes in tfp
  h = dict([[tfp[0][i], i] for i in range(len(tfp[0]))])

  # load global parameters used for c_stats
  c_livetime = tfp[1][h['livetime']]
  c_ngwtrg = tfp[1][h['#gwtrg']]
  ngwtrg_vtd = tfp[1][h['#gwtrg']] - tfp[len(tfp) - 1][h['#gwtrg']] + tfp[len(tfp) -1][h['vact']]

  # instantiate counters for c_stats
  c_dsec = 0.0
  c_csec = 0.0
  c_vact = 0.0

  # instantiate the storage device and the index counter
  gwtrg_vtd_tcent = [0]*ngwtrg_vtd
  tcentidx = 0

  # iterate over tfp and fill in gwtrg_vtd_tcent
  for lineidx in range(len(tfp)):
    line = tfp[lineidx]
    if line[0] == 'livetime':
      pass
    else:
      # compute cumulative quantities and lists to be stored under gwtcent
      c_dsec += line[h['dsec']]
      c_csec += line[h['csec']]
      c_vact += line[h['vact']]
      vconfig = [line[h['vchan']], line[h['vthr']], line[h['vwin']]]
      vstats = [line[h['livetime']], line[h['#gwtrg']], line[h['dsec']], line[h['csec']], line[h['vact']], line[h['vsig']], c_livetime, c_ngwtrg, c_dsec, c_csec, c_vact, lineidx]
      gwtrg_vtd = line[h['gwtrg_vetoed']]
      # iterate through trg's and fill in gwtrg_vtd_tcent
      for trg in gwtrg_vtd:
        gwtrg_vtd_tcent[tcentidx] = [trg[col_kw['tcent']], vconfig, vstats]
        tcentidx += 1

  # sort gwtrg_vtd_tcent by tcent for ease of use by the caller
  return sorted(gwtrg_vtd_tcent, key=lambda gwtrg: gwtrg[0])


  
