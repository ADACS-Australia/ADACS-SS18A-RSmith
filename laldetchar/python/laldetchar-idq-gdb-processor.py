# Copyright (C) 2013 Lindy Blackburn, Reed Essick and Ruslan Vaulin
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

import os
import signal
import optparse
import subprocess
import argparse
import sys

import ConfigParser

from laldetchar.idq import idq
from laldetchar import git_version

__author__ = \
    'Lindy Blackburn (<lindy.blackburn@ligo.org>), Reed Essick (<reed.essick@ligo.org>), Ruslan Vaulin (<ruslan.vaulin@ligo.org>)'
__version__ = git_version.id
__date__ = git_version.date



def getProcessData():
    ps = idq.fork(['ps', 'aux', '-ww'], stdout=subprocess.PIPE).communicate()[0]
    processes = ps.split('\n')
    # this specifies the number of splits, so the splitted lines
    # will have (nfields+1) elements
    nfields = len(processes[0].split()) - 1
    retval = []
    for row in processes[1:]:
	    # skip row if it is empty
        if row: retval.append(row.split(None, nfields))
    return retval


description = \
    """This program manages lvalert listener for idq pipeline. It can start, check status or stop listening process(es). The logic is similar to that of gdb_processor."""

parser = argparse.ArgumentParser(usage='%%prog [options] [arguments]', description=description)
parser.add_argument(
    'command',
    choices = ['start', 'status', 'stop'],
    help='start, return status or stop idq lvalert listening processes and all related tasks.',
    )
parser.add_argument(
    '--version',
    action='version',
    version= 'Name: %%prog\n%s'
    % git_version.verbose_msg,
    help = 'show code version'
    )

parser.add_argument(
    '--config',
    help = "the config file for the LVAlert process",
    default = "idq_gdb.ini",
    )

parser.add_argument(
    '--dont-wait',
    help = "do not wait for jobs to finish before launching new jobs",
    default=False,
    action="store_true"
    )

args = parser.parse_args()

# if command is start, then run launch lvalert listener.
if args.command == 'start':

    ### get required data from config 
    config = ConfigParser.SafeConfigParser()
    config.read(args.config)

    server = config.get("lvalert_listener", "server")
    username = config.get("lvalert_listener", "username")
    lvalert_config = config.get("lvalert_listener", "lvalert_config")

    ### write lvalert_config file based on information from config
    lvalert_config_obj = ConfigParser.SafeConfigParser()

    for node in config.options("nodes"): ### iterate over nodes listed in config
        lvalert_config_obj.add_section( node )
        lvalert_config_obj.set( node, "executable", config.get("lvalert_listener", "executable") )
#        lvalert_config_obj.set( node, "plotting_time_before", config.getfloat(node, "plotting_time_before") )
#        lvalert_config_obj.set( node, "plotting_time_after", config.getfloat(node, "plotting_time_after") )
#        lvalert_config_obj.set( node, "time_before", config.getfloat(node, "time_before") )
#        lvalert_config_obj.set( node, "time_after", config.getfloat(node, "time_after") )
#        lvalert_config_obj.set( node, "max_wait", config.getfloat(node, "max_wait") )
#        lvalert_config_obj.set( node, "delay", config.getfloat(node, "delay") )

    ### add heartbeat response
    if config.has_section('heartbeat'):
        node = config.get('heartbeat', 'node')
        lvalert_config_obj.add_section( node )
        lvalert_config_obj.set( node, "executable", config.get('heartbeat', "executable") )

    lvalert_config_file = open(lvalert_config, "w")
    lvalert_config_obj.write(lvalert_config_file) ### write new config to disk
    lvalert_config_file.close()

    print "Launching lvalert listener ..."
    lvalert_launch_command = [
                'nohup',
                'lvalert_listen',
                '--username',
                username,
                '--config-file',
                lvalert_config,
                '--server',
                server
                ]
    if args.dont_wait:
        lvalert_launch_command.insert(2, "--dont-wait" )
    if config.has_option("lvalert_listener","resource_name"):
        lvalert_launch_command += ["-r", config.get("lvalert_listener","resource_name")]

    pid = idq.fork(lvalert_launch_command, stdout=open('lvalert_listen.out', 'a')).pid

    print "lvalert_listen is launched with process id " + str(pid)
    sys.exit(0)

# if command is status, get currently running idq lvalert listening process(es)and print basic status information.
elif args.command == 'status':

    # determin the current user name
    user = os.getenv('USER')

    # run ps and parse its output
    pstats = getProcessData()

    # print all lvalert_listen processes run by the current user
    print "lvalert_listen process data:"
    print "USER              PID       %CPU        %MEM       VSZ        RSS        TTY       STAT      START       TIME   COMMAND"

    for ps in pstats:
        if (not len(ps) >= 1): continue
        if (ps[0] == user) and ('lvalert_listen' in ps[10]):
            print "%-10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s  %s" % (ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10])



# if command is stop, get the list of currently running lv alert listening process(es) and terminate them.
elif args.command == 'stop':

    # determin the current user name
    user = os.getenv('USER')

    # run ps and parse its output
    pstats = getProcessData()

    # kill lvalert_listen processes run by the current user
    for ps in pstats:
        if (not len(ps) >= 1): continue
        if (ps[0] == user) and ('lvalert_listen' in ps[10]):
            print "USER              PID       %CPU        %MEM       VSZ        RSS        TTY       STAT      START       TIME   COMMAND"
            print "%-10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s %10.10s  %s" % (ps[0], ps[1], ps[2], ps[3], ps[4], ps[5], ps[6], ps[7], ps[8], ps[9], ps[10])
            exterminate = raw_input('Do you want to kill this process?? [y/n]')
            if exterminate == 'y' or exterminate == 'yes':
                print "killing process " + ps[1]
                os.kill(int(ps[1]), signal.SIGQUIT)
            else:
                print "skiping this process"
			
			


							
							
