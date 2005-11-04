"""
The LDBDServer module provides an API for responding to request from the
LDBDClient by connecting to the DB2 database.

This module requires U{pyGlobus<http://www-itg.lbl.gov/gtg/projects/pyGlobus/>}.

$Id$
"""

__version__ = '$Revision$'[11:-2]

import sys
import re
import types
import pyRXP
import exceptions
import socket
import SocketServer
import cPickle
from glue import ldbd
import rlsClient

def initialize(configuration,log):
  # define the global variables used by the server
  global logger, max_bytes, xmlparser, lwtparser, dbobj, rls
  global dmt_proc_dict, dmt_seg_def_dict, indices
  
  # initialize the logger
  logger = log
  logger.info("Initializing server module %s" % __name__ )
  
  # initialize the database hash table
  dbobj = ldbd.LIGOMetadataDatabase(configuration['dbname'])
  max_bytes = configuration['max_client_byte_string']

  # create the xml and ligolw parsers
  xmlparser = pyRXP.Parser()
  lwtparser = ldbd.LIGOLwParser()

  # open a connection to the rls server
  rls_server = configuration['rls']
  cert = configuration['certfile']
  key = configuration['keyfile']
  try:
    rls = rlsClient.RlsClient(rls_server,cert,key)
  except:
    rls = None

  # initialize dictionaries for the dmt processes and segments definers
  dmt_proc_dict = {}
  dmt_seg_def_dict = {}
  indices = xrange(sys.maxint)

def shutdown():
  global logger, max_bytes, xmlparser, lwtparser, dbobj, rls
  global dmt_proc_dict, dmt_seg_def_dict
  logger.info("Shutting down server module %s" % __name__ )
  if rls:
    del rls
  del lwtparser
  del xmlparser
  del dbobj
  del dmt_proc_dict
  del dmt_seg_def_dict

class ServerHandlerException(exceptions.Exception):
  """
  Class representing exceptions within the ServerHandler class.
  """
  def __init__(self, args=None):
    """
    Initialize an instance.

    @param args: 

    @return: Instance of class ServerHandlerException
    """
    self.args = args
        
class ServerHandler(SocketServer.BaseRequestHandler):
  """
  An instance of this class is created to service each request of the server.
  """
  def handle(self):
    """
    This method does all the work of servicing a request to the server. See
    the documentation for the standard module SocketServer.

    The input from the socket is parsed for the method with the remaining
    strings stripped of null bytes passed to the method in a list.

    There are no parameters. When the instance of the class is created to
    process the request all necessary information is made attributes of the
    class instance.

    @return: None
    """
    global logger
    global max_bytes

    logger.debug("handle method of %s class called" % __name__)

    # mapping of ldbdd RPC protocol names to methods of this class
    methodDict = {
      'PING' : self.ping,
      'QUERY' : self.query,
      'INSERT' : self.insert,
      'INSERTMAP' : self.insertmap,
      'INSERTDMT' : self.insertdmt
    }

    try:
      # from the socket object create a file object
      self.sfile = self.request.makefile("rw")
      f = self.sfile

      # read all of the input up to limited number of bytes
      input = f.read(size=max_bytes,waitForBytes=2)

      # try 10 more times if we don't have a null byte at the end
      readnum = 0
      while input[-1] != '\0' and readnum < 10:
        input += f.read(size=max_bytes,waitForBytes=2)
        readnum = readnum + 1

      # the format should be a method string, followed by a null byte
      # followed by the arguments to the method encoded as null
      # terminated strings

      # check if the last byte is a null byte
      if input[-1] != '\0':
        logger.info("Bad input on socket: %s" % input)
        raise ServerHandlerException, "Last byte of input is not null byte"
    except Exception, e:
      logger.error("Error reading input on socket: %s" %  e)
      return

    logger.debug("Input on socket: %s" % input[0:-1])

    try:
      # parse out the method and arguments 
      stringList = input.split('\0')
      methodString = stringList[0]
      argStringList = stringList[1:-1]
                        
    except Exception, e:
      logger.error("Error parsing method and argument string: %s" % e)

      msg = "ERROR LDBDServer Error: " + \
        "Error parsing method and argument string: %s" % e
      self.__reply__(1, msg)
      return
                
    try:
      # look up method in dictionary
      method = methodDict[methodString]
    except Exception, e:
      msg = "Error converting method string %s to method call: %s" % \
        (methodString, e)
      logger.error(msg)
                        
      self.__reply__(1, msg)
      return

    try:
      # call the method requested with the rest of strings as input
      result = method(argStringList) 
      self.__reply__( result[0], result[1] )
    except Exception, e:
      logger.error("Error while calling method %s: %s" % (methodString, e))

    return
        
  def __reply__(self, code, msg):
    """
    Format and send a reply back down the socket to the client. The file
    representing the socket is closed at the end of this method.

    @param code: integer representing the error code with 0 for success
                
    @param msg: object to be passed back to the client, either a string
    or a list of items that can be represented by strings
                        
    @return: None
    """
    f = self.sfile
    reply = "%d\0%s\0" % (code, msg)
    f.write(reply)

    # close the file associated with the socket
    f.close()

  def ping(self, arg):
    """
    Bounce back alive statment. Corresponds to the PING method in the
    ldbdd RPC protocol.

    @param arg: list (perhaps empty) of strings representing message sent
      by client to server

    @return: None
    """
    global logger

    logger.debug("Method ping called")
    try:
      hostname = socket.getfqdn()
      msg = "%s at %s is alive" % (__name__, hostname)
    except Exception, e:
      msg = "%s is alive" % __name__

    return (0, msg)

  def query(self, arg):
    """
    Execute an SQL query on the database and return the result as LIGO_LW XML

    @param arg: a text string containing an SQL query to be executed

    @return: None
    """
    global logger
    global xmlparser, lwtparser, dbobj

    # get the query string and log it
    querystr = arg[0]
    logger.debug("Method query called with %s" % querystr)

    # assume failure
    code = 1

    try:
      # create a ligo metadata object
      ligomd = ldbd.LIGOMetadata(xmlparser,lwtparser,dbobj)

      # execute the query
      rowcount = ligomd.select(querystr)

      # convert the result to xml
      result = ligomd.xml()

      logger.debug("Method query: %d rows returned" % rowcount)
      code = 0
    except Exception, e:
      result = ("Error querying metadata database: %s" % e)
      logger.error(result)

    try:
      del ligomd
    except Exception, e:
      logger.error(
        "Error deleting metadata object in method query: %s" % e)

    return (code,result)

  def insert(self, arg):
    """
    Insert some LIGO_LW xml data in the metadata database

    @param arg: a text string containing an SQL query to be executed

    @return: None
    """
    global logger
    global xmlparser, lwtparser, dbobj

    logger.debug("Method insert called")

    # assume failure
    code = 1

    try:
      # capture the remote users DN for insertion into the database
      cred = self.request.get_delegated_credential()
      remote_dn = cred.inquire_cred()[1].display()

      # create a ligo metadata object
      ligomd = ldbd.LIGOMetadata(xmlparser,lwtparser,dbobj)

      # parse the input string into a metadata object
      ligomd.parse(arg[0])

      # add a gridcert table to this request containing the users dn
      ligomd.set_dn(remote_dn)

      # insert the metadata into the database
      result = str(ligomd.insert())

      logger.info("Method insert: %s rows affected by insert" % result)
      code = 0
    except Exception, e:
      result = ("Error inserting metadata into database: %s" % e)
      logger.error(result)

    try:
      del ligomd
    except Exception, e:
      logger.error(
        "Error deleting metadata object in method insert: %s" % e)

    return (code,result)

  def insertmap(self, arg):
    """
    Insert some LIGO_LW xml data in the metadata database with an LFN to
    PFN mapping inserted into the RLS database.

    @param arg: a text string containing an SQL query to be executed

    @return: None
    """
    global logger
    global xmlparser, lwtparser, dbobj, rls

    logger.debug("Method insertmap called")

    if not rls:
      msg = "server is not initialized for RLS connections"
      logger.error(msg)
      return (1, msg)

    # assume failure
    code = 1

    try:
      # unpickle the PFN/LFN mappings from the client
      lfnpfn_dict = cPickle.loads(arg[1])
      if not isinstance(lfnpfn_dict, dict):
        raise ServerHandlerException, \
          "LFN/PFN mapping from client is not dictionary"

      # capture the remote users DN for insertion into the database
      cred = self.request.get_delegated_credential()
      remote_dn = cred.inquire_cred()[1].display()

      # create a ligo metadata object
      ligomd = ldbd.LIGOMetadata(xmlparser,lwtparser,dbobj)

      # parse the input string into a metadata object
      ligomd.parse(arg[0])

      # add a gridcert table to this request containing the users dn
      ligomd.set_dn(remote_dn)

      # add the lfns to the metadata insert to populate the lfn table
      for lfn in lfnpfn_dict.keys():
        ligomd.add_lfn(lfn)

      # insert the metadata into the database
      result = str(ligomd.insert())
      logger.info("Method insert: %s rows affected by insert" % result)

      # insert the PFN/LFN mappings into the RLS
      for lfn in lfnpfn_dict.keys():
        pfns = lfnpfn_dict[lfn]
        if not isinstance( pfns, types.ListType ):
          raise ServerHandlerException, \
            "PFN must be a single string or a list of PFNs"
        rls.lrc_create_lfn( lfn, pfns[0] )
        for pfn in pfns[1:len(pfns)]:
          rls.lrc_add( lfn, pfn )
          
      logger.info("Method insertmap: insert LFN mappings for %s" % 
        str(lfnpfn_dict.keys()))
      code = 0

    except Exception, e:
      result = ("Error inserting LFN/PFN mapping into RLS: %s" % e)
      logger.error(result)
      return (code,result)

    return (code,result)

  def insertdmt(self, arg):
    """
    Insert LIGO_LW xml data from the DMT in the metadata database. For
    DMT inserts, we need to check for existing process_id and
    segment_definer_id rows and change the contents of the table to be
    inserted accordingly. We must also update the end_time of any 
    existing entries in the process table.

    @param arg: a text string containing an SQL query to be executed

    @return: None
    """
    global logger
    global xmlparser, lwtparser, dbobj
    global dmt_proc_dict, dmt_seg_def_dict, indices
    proc_key = {}
    known_proc = {}
    seg_def_key = {}

    msg = "Method dmtinsert called. Known processes %s, " % str(dmt_proc_dict)
    msg += "known segment_definers %s" % str(dmt_seg_def_dict)
    logger.debug(msg)

    # assume failure
    code = 1

    try:
      # capture the remote users DN for insertion into the database
      cred = self.request.get_delegated_credential()
      remote_dn = cred.inquire_cred()[1].display().strip()

      # create a ligo metadata object
      ligomd = ldbd.LIGOMetadata(xmlparser,lwtparser,dbobj)

      # parse the input string into a metadata object
      ligomd.parse(arg[0])

      # determine the local creator_db number
      sql = "SELECT DEFAULT FROM SYSCAT.COLUMNS WHERE "
      sql += "TABNAME = 'PROCESS' AND COLNAME = 'CREATOR_DB'"
      ligomd.curs.execute(sql)
      creator_db = ligomd.curs.fetchone()[0]

      # determine the locations of columns we need in the process table
      process_cols = ligomd.table['process']['orderedcol']
      node_col = process_cols.index('node')
      prog_col = process_cols.index('program')
      upid_col = process_cols.index('unix_procid')
      start_col = process_cols.index('start_time')
      end_col = process_cols.index('end_time')
      pid_col = process_cols.index('process_id')

      # determine and remove known entries from the process table
      rmv_idx = []
      for (row,row_idx) in zip(ligomd.table['process']['stream'],indices):
        uniq_proc = (row[node_col],row[prog_col],row[upid_col],row[start_col])
        logger.debug("Checking for process row with key %s" % str(uniq_proc))
        try:
          proc_key[row[pid_col]] = dmt_proc_dict[uniq_proc]
          known_proc[dmt_proc_dict[uniq_proc]] = row[end_col]
          logger.debug("removing known process row for key %s" % str(uniq_proc))
          rmv_idx.append(row_idx)
        except KeyError:
          # we know nothing about this process, so query the database
          sql = "SELECT process_id FROM process WHERE "
          sql += "creator_db = " + str(creator_db) + " AND "
          sql += "node = '" + row[node_col] + "' AND "
          sql += "program = '" + row[prog_col] + "' AND "
          sql += "unix_procid = " + str(row[upid_col]) + " AND "
          sql += "start_time = " + str(row[start_col])
          ligomd.curs.execute(sql)
          db_proc_ids = ligomd.curs.fetchall()
          if len(db_proc_ids) == 0:
            # this is a new process with no existing entry
            dmt_proc_dict[uniq_proc] = row[pid_col]
          elif len(db_proc_ids) == 1:
            # the process_id exists in the database so use that insted
            logger.debug("process row for key %s exists in database" 
              % str(uniq_proc))
            dmt_proc_dict[uniq_proc] = db_proc_ids[0][0]
            proc_key[row[pid_col]] = dmt_proc_dict[uniq_proc]
            known_proc[dmt_proc_dict[uniq_proc]] = row[end_col]
            logger.debug("removing process row for key %s" % str(uniq_proc))
            rmv_idx.append(row_idx)
          else:
            # multiple entries for this process, needs human assistance
            raise ServerHandlerException, "multiple entries for dmt process"

      # delete the duplicate processs rows and clear the table if necessary
      newstream = []
      for row,row_idx in zip(ligomd.table['process']['stream'],indices):
        try:
          rmv_idx.index(row_idx)
        except ValueError:
          newstream.append(row)
      ligomd.table['process']['stream'] = newstream
      if len(ligomd.table['process']['stream']) == 0:
        del ligomd.table['process']

      # turn the known process_id binary for this insert into ascii
      for pid in known_proc.keys():
        pid_str = "x'"
        for ch in pid:
          pid_str += "%02x" % ord(ch)
        pid_str += "'"
        known_proc[pid] = (pid_str, known_proc[pid])

      # check that we have permission to update known process_id entries
      for pid in known_proc.keys():
        sql = "SELECT dn FROM gridcert WHERE process_id = " + known_proc[pid][0]
        sql += "AND creator_db = " + str(creator_db)
        ligomd.curs.execute(sql)
        dn_db = ligomd.curs.fetchone()
        if not dn_db:
          msg = "Could not find DN for process %s" % known_proc[pid][0]
          raise ServerHandlerException, msg
        else:
          dn = dn_db[0].strip()
        if remote_dn != dn:
          msg = "%s does not have permission to update row entries" % remote_dn
          msg += " created by %s (process_id %s)" % (dn, known_proc[pid][0])
          raise ServerHandlerException, msg
        else:
          logger.debug('"%s" updating process_id %s' % (dn, known_proc[pid][0]))

      # add a gridcert table to this request containing the users dn
      ligomd.set_dn(remote_dn)

      # determine the locations of columns we need in the segment_definer table
      seg_def_cols = ligomd.table['segment_definer']['orderedcol']
      run_col = seg_def_cols.index('run')
      ifos_col = seg_def_cols.index('ifos')
      name_col = seg_def_cols.index('name')
      vers_col = seg_def_cols.index('version')
      sdid_col = seg_def_cols.index('segment_def_id')

      # determine and remove known entries in the segment_definer table
      rmv_idx = []
      for row,row_idx in zip(ligomd.table['segment_definer']['stream'],indices):
        uniq_def = (row[run_col],row[ifos_col],row[name_col],row[vers_col])
        logger.debug("Checking for segment_definer row with key %s" 
          % str(uniq_def))
        try:
          seg_def_key[row[sdid_col]] = dmt_seg_def_dict[uniq_def]
          logger.debug("removing known segment_definer row for key %s" 
            % str(uniq_def))
          rmv_idx.append(row_idx)
        except KeyError:
          # we know nothing about this segment_definer, so query the database
          sql = "SELECT segment_def_id FROM segment_definer WHERE "
          sql += "creator_db = " + str(creator_db) + " AND "
          sql += "run = '" + row[run_col] + "' AND "
          sql += "ifos = '" + row[ifos_col] + "' AND "
          sql += "name = '" + row[name_col] + "' AND "
          sql += "version = " + str(row[vers_col])
          ligomd.curs.execute(sql)
          db_seg_def_id = ligomd.curs.fetchall()
          if len(db_seg_def_id) == 0:
            # this is a new segment_defintion with no existing entry
            dmt_seg_def_dict[uniq_def] = row[sdid_col]
          else:
            logger.debug("segment_definer row for key %s exists in database" 
              % str(uniq_def))
            dmt_seg_def_dict[uniq_def] = db_seg_def_id[0][0]
            seg_def_key[row[sdid_col]] = dmt_seg_def_dict[uniq_def]
            logger.debug("removing segment_definer row for key %s" 
              % str(uniq_def))
            rmv_idx.append(row_idx)

      # delete the necessary rows. if the table is empty, delete it
      newstream = []
      for row,row_idx in zip(ligomd.table['segment_definer']['stream'],indices):
        try:
          rmv_idx.index(row_idx)
        except ValueError:
          newstream.append(row)
      ligomd.table['segment_definer']['stream'] = newstream
      if len(ligomd.table['segment_definer']['stream']) == 0:
        del ligomd.table['segment_definer']

      # now update the values in the xml with the values we know about
      for tabname in ligomd.table.keys():
        table = ligomd.table[tabname]
        if tabname == 'process':
          # we do nothing to the process table
          pass
        elif tabname == 'segment_def_map':
          # we need to update the process_id and the segment_def_id columns
          pid_col = table['orderedcol'].index('process_id')
          sdid_col = table['orderedcol'].index('segment_def_id')
          row_idx = 0
          for row in table['stream']:
            try:
              repl_pid = proc_key[row[pid_col]]
            except KeyError:
              repl_pid = row[pid_col]
            try:
              repl_sdid = seg_def_key[row[sdid_col]]
            except KeyError:
              repl_sdid = row[sdid_col]
            row = list(row)
            row[pid_col] = repl_pid
            row[sdid_col] = repl_sdid
            table['stream'][row_idx] = tuple(row)
            row_idx += 1
        else:
          # we just need to update the process_id column
          pid_col = table['orderedcol'].index('process_id')
          row_idx = 0
          for row in table['stream']:
            try:
              repl_pid = proc_key[row[pid_col]]
              row = list(row)
              row[pid_col] = repl_pid
              table['stream'][row_idx] = tuple(row)
            except KeyError:
              pass
            row_idx += 1

      # insert the metadata into the database
      result = str(ligomd.insert())

      # update the end time of known processes in the process table
      for pid in known_proc.keys():
        sql = "UPDATE process SET end_time = " + str(known_proc[pid][1])
        sql += " WHERE process_id = " + known_proc[pid][0]
        ligomd.curs.execute(sql)
      ligomd.dbcon.commit()

      logger.info("Method insert: %s rows affected by insert" % result)
      code = 0
    except Exception, e:
      result = ("Error inserting metadata into database: %s" % e)
      logger.error(result)

    try:
      del ligomd
      del known_proc
      del seg_def_key
      del proc_key
    except Exception, e:
      logger.error(
        "Error deleting metadata object in method insert: %s" % e)

    return (code,result)

