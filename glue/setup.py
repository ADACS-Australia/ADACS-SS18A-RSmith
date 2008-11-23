# $Id$
# 
# setup script for glue

import os, sys

try:
  from sys import version_info
except:
  print >> sys.stderr, "Unable to determine the python version"
  print >> sys.stderr, "Please check that your python version is >= 2.4"
  sys.exit(1)

if version_info < (2, 4):
  print >> sys.stderr, "Your python version " + str(version_info) + " appears to be less than 2.4"
  print >> sys.stderr, "Please check that your python version is >= 2.4"
  print >> sys.stderr, "Glue requires at least version 2.4"
  sys.exit(1)

from distutils.core import setup, Extension
from distutils.command import install
from distutils.command import sdist
from distutils import log

ver = "1.18"

def remove_root(path,root):
  if root:
    return os.path.normpath(path).replace(os.path.normpath(root),"")
  else:
    return os.path.normpath(path)

class glue_install(install.install):
  def run(self):

    # create the user env scripts
    if self.install_purelib == self.install_platlib:
      glue_pythonpath = self.install_purelib
    else:
      glue_pythonpath = self.install_platlib + ":" + self.install_purelib

    glue_prefix = remove_root(self.prefix,self.root)
    glue_install_scripts = remove_root(self.install_scripts,self.root)
    glue_pythonpath = remove_root(glue_pythonpath,self.root)
    glue_install_platlib = remove_root(self.install_platlib,self.root)
    
    log.info("creating glue-user-env.sh script")
    env_file = open(os.path.join('etc','glue-user-env.sh'),'w')
    print >> env_file, "# Source this file to access GLUE"
    print >> env_file, "GLUE_PREFIX=" + glue_prefix
    print >> env_file, "export GLUE_PREFIX"
    print >> env_file, "PATH=" + glue_install_scripts + ":${PATH}"
    print >> env_file, "PYTHONPATH=" + glue_pythonpath + ":${PYTHONPATH}"
    print >> env_file, "LD_LIBRARY_PATH=" + glue_install_platlib + ":${LD_LIBRARY_PATH}"
    print >> env_file, "DYLD_LIBRARY_PATH=" + glue_install_platlib + ":${DYLD_LIBRARY_PATH}"
    print >> env_file, "export PATH PYTHONPATH LD_LIBRARY_PATH DYLD_LIBRARY_PATH"
    env_file.close()

    log.info("creating glue-user-env.csh script")
    env_file = open(os.path.join('etc','glue-user-env.csh'),'w')
    print >> env_file, "# Source this file to access GLUE"
    print >> env_file, "setenv GLUE_PREFIX " + glue_prefix
    print >> env_file, "setenv PATH " + glue_install_scripts + ":${PATH}"
    print >> env_file, "if ( $?PYTHONPATH ) then"
    print >> env_file, "  setenv PYTHONPATH " + glue_pythonpath + ":${PYTHONPATH}"
    print >> env_file, "else"
    print >> env_file, "  setenv PYTHONPATH " + glue_pythonpath
    print >> env_file, "endif"
    print >> env_file, "if ( $?LD_LIBRARY_PATH ) then"
    print >> env_file, "  setenv LD_LIBRARY_PATH " + glue_install_platlib + ":${LD_LIBRARY_PATH}"
    print >> env_file, "else"
    print >> env_file, "  setenv LD_LIBRARY_PATH " + glue_install_platlib
    print >> env_file, "endif"
    print >> env_file, "if ( $?DYLD_LIBRARY_PATH ) then"
    print >> env_file, "  setenv DYLD_LIBRARY_PATH " + glue_install_platlib + ":${DYLD_LIBRARY_PATH}"
    print >> env_file, "else"
    print >> env_file, "  setenv DYLD_LIBRARY_PATH " + glue_install_platlib
    print >> env_file, "endif"
    env_file.close()

    # now run the installer
    install.install.run(self)

class glue_sdist(sdist.sdist):
  def run(self):
    # remove the automatically generated user env scripts
    for script in [ 'glue-user-env.sh', 'glue-user-env.csh' ]:
      log.info( 'removing ' + script )
      try:
        os.unlink(os.path.join('etc',script))
      except:
        pass

    # now run sdist
    sdist.sdist.run(self)

setup(
  name = "glue",
  version = ver,
  author = "Duncan Brown",
  author_email = "dbrown@ligo.caltech.edu",
  description = "Grid LSC User Engine",
  url = "http://www.lsc-group.phys.uwm.edu/daswg/",
  license = 'See file LICENSE',
  packages = [ 'glue', 'glue.lars', 'glue.lars.cli', 'glue.lars.util', 'glue.ligolw', 'glue.ligolw.utils' ],
  cmdclass = { 'install' : glue_install, 'sdist' : glue_sdist },
  py_modules = [ 'glue.segfindserver.segments_1_7.segments' ],
  ext_modules = [
    Extension(
      "glue.ligolw.tokenizer",
      [
        "glue/ligolw/tokenizer.c",
        "glue/ligolw/tokenizer.Tokenizer.c",
        "glue/ligolw/tokenizer.RowBuilder.c",
        "glue/ligolw/tokenizer.RowDumper.c"
      ],
      include_dirs = [ "glue/ligolw" ]
    ),
    Extension(
      "glue.ligolw.__ilwd",
      [
        "glue/ligolw/ilwd.c"
      ],
      include_dirs = [ "glue/ligolw" ]
    ),
    Extension(
      "glue.__segments",
      [
        "src/segments/segments.c",
        "src/segments/infinity.c",
        "src/segments/segment.c",
        "src/segments/segmentlist.c"
      ],
      include_dirs = [ "src/segments" ]
    )
  ],
  scripts = [
    os.path.join('bin','LSCdataFind'),
    os.path.join('bin','LSCsegFind'),
    os.path.join('bin','LSCsegFindDev'),
    os.path.join('bin','LSCfileAdd'),
    os.path.join('bin','lars'),
    os.path.join('bin','lars_add'),
    os.path.join('bin','lars_search'),
    os.path.join('bin','ldbdc'),
    os.path.join('bin','ldg_submit_dax'),
    os.path.join('bin','dmtdq_seg_insert'),
    os.path.join('bin','ligolw_add'),
    os.path.join('bin','ligolw_cut'),
    os.path.join('bin','ligolw_burst2mon'),
    os.path.join('bin','ligolw_inspiral2mon'),
    os.path.join('bin','ligolw_print'),
    os.path.join('bin','ligolw_sqlite'),
    os.path.join('sbin','ldbdd'),
    os.path.join('sbin','segpagegen'),
    os.path.join('sbin','LSCdqInsert'),
    os.path.join('sbin','publishstatefromfile'),
    os.path.join('sbin','bulkpublishstate'), ],
  data_files = [
    (
      'etc',
      [
        os.path.join('etc','ldg-sites.xml'),
        os.path.join('etc','pegasus-properties.bundle'),
        os.path.join('etc','glue-user-env.sh'),
        os.path.join('etc','glue-user-env.csh'),
        os.path.join('etc','lscsegfindserver.ini'),
        os.path.join('etc','lscsegfindserverdev.ini'),
        os.path.join('etc','segpagegen.ini'),
        os.path.join('etc','segpagegen_S5.ini'),
        os.path.join('etc','segpagegen_A5.ini'),
        os.path.join('etc','segpagegen_S5.sh'),
        os.path.join('etc','segpagegen_A5.sh'),
        os.path.join('etc','ldbdserver.ini'),
        os.path.join('etc','ligolw_dtd.txt')
      ]
    )
  ]
)
